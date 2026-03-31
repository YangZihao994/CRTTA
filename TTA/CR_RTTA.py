import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch.cuda.amp import autocast, GradScaler


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence (batch-wise). Returns (B,)."""
    eps = 1e-9
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp(eps).log() - m.clamp(eps).log())).sum(-1)
    kl_qm = (q * (q.clamp(eps).log() - m.clamp(eps).log())).sum(-1)
    return 0.5 * (kl_pm + kl_qm).clamp(min=0.0)


def sharpen(p: torch.Tensor, T: float = 0.5) -> torch.Tensor:
    """Sharpen a probability distribution with temperature T."""
    sharp = p.pow(1.0 / T)
    return sharp / sharp.sum(dim=-1, keepdim=True).clamp(min=1e-9)


# ─────────────────────────────────────────────
#  Uncertainty Decomposition
# ─────────────────────────────────────────────

def compute_uncertainty(
    pm: torch.Tensor,
    pm_aug_list: list,
    pn_list: list,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
):
    """
    Compute holistic unreliability score U_m for one modality.

    pm           : (B, C)  softmax probs for modality m
    pm_aug_list  : list of (B, C) augmented probs (K items)
    pn_list      : list of (B, C) other-modality probs (M-1 items)
    Returns: H_m, C_m, A_m, U_m — each (B,)
    """
    B, C = pm.shape
    eps = 1e-9

    # 1. Normalised predictive entropy
    H_m = -(pm * pm.clamp(eps).log()).sum(-1) / math.log(C)   # (B,)

    # 2. Multi-view consistency
    K = len(pm_aug_list)
    if K > 0:
        js_sum = sum(js_divergence(pm, pk) for pk in pm_aug_list)
        C_m = 1.0 - js_sum / K
    else:
        C_m = torch.ones(B, device=pm.device)

    # 3. Cross-modal agreement
    M_1 = len(pn_list)
    if M_1 > 0:
        js_sum = sum(js_divergence(pm, pn) for pn in pn_list)
        A_m = 1.0 - js_sum / M_1
    else:
        A_m = torch.ones(B, device=pm.device)

    U_m = alpha * H_m + beta * (1.0 - C_m) + gamma * (1.0 - A_m)
    return H_m, C_m, A_m, U_m


# ─────────────────────────────────────────────
#  Feature-Aware Residual Side-Car
# ─────────────────────────────────────────────

class SideCar(nn.Module):
    """
    Lightweight MLP that ingests per-modality logits, projected features,
    and uncertainty scores, and outputs a residual correction Δz.
    """

    def __init__(
        self,
        n_class: int,
        feat_dim: int = 768,
        proj_dim: int = 128,
        n_modalities: int = 2,
    ):
        super().__init__()
        self.W_proj = nn.Linear(feat_dim, proj_dim)

        # concat: logits (n_class × M)  + features (proj_dim × M)  + U (M)
        in_dim = n_class * n_modalities + proj_dim * n_modalities + n_modalities
        hidden = max(in_dim * 2, 512)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, n_class),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # zero-initialise the last layer so Δz ≈ 0 at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        logits_list: list,   # [z_a (B,C), z_v (B,C)]
        feat_list: list,     # [f_a (B,D), f_v (B,D)]
        u_list: list,        # [U_a (B,), U_v (B,)]
    ) -> torch.Tensor:
        proj = [F.relu(self.W_proj(f)) for f in feat_list]
        u_cols = [u.unsqueeze(-1) for u in u_list]
        x = torch.cat(logits_list + proj + u_cols, dim=-1)
        return self.mlp(x)


# ─────────────────────────────────────────────
#  CR-RTTA  (main module)
# ─────────────────────────────────────────────

class CR_RTTA(nn.Module):
    """
    Architecture-Agnostic Multi-Modal Test-Time Adaptation.

    Backbone is kept fully frozen.
    Only SideCar (ϕ) and its W_proj are updated per batch.
    """

    def __init__(self, model, optimizer, device, args, steps: int = 1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.args = args
        self.device = device
        self.scaler = GradScaler()

        # ── uncertainty hyper-params ──────────────────────────────
        self.alpha   = getattr(args, "unc_alpha",   1.0)
        self.beta    = getattr(args, "unc_beta",    1.0)
        self.gamma_u = getattr(args, "unc_gamma",   1.0)
        self.tau     = getattr(args, "tau",         1.0)   # temperature for trust weights
        self.mu      = getattr(args, "mu",          0.9)   # EMA decay
        self.theta   = getattr(args, "theta",       0.5)   # confidence gate
        self.K_aug   = getattr(args, "K_aug",       4)     # augmentation views
        self.noise_std = getattr(args, "aug_noise",  0.01)

        # ── loss hyper-params ─────────────────────────────────────
        self.lambda_max = getattr(args, "lambda_max", 1.0)
        self.lambda2    = getattr(args, "lambda2",    1.0)
        self.lambda3    = getattr(args, "lambda3",    1.0)
        self.lambda4    = getattr(args, "lambda4",    0.01)
        self.eta        = getattr(args, "eta",        0.01)  # warm-up growth rate

        # ── EMA buffer & step counter ─────────────────────────────
        self.register_buffer("p_ema", torch.zeros(1))
        self.p_ema_init = False
        self.t = 0  # global adaptation step

        # ── Side-Car (only trainable component) ───────────────────
        n_class  = args.n_class
        feat_dim = getattr(args, "feat_dim",  768)
        proj_dim = getattr(args, "proj_dim",  128)
        self.side_car = SideCar(n_class, feat_dim, proj_dim, n_modalities=2).to(device)

    # ─────────────────────────────────────────
    def forward(self, x, adapt_flag: bool):
        a_input, v_input = x

        for _ in range(self.steps):
            if adapt_flag:
                out, loss = self._forward_and_adapt(a_input, v_input)
            else:
                out, loss = self._forward_only(a_input, v_input)

        return out, loss

    # ─────────────────────────────────────────
    @torch.no_grad()
    def _forward_only(self, a, v):
        with autocast():
            res = self.model.module.forward_eval_crrtta(a=a, v=v, mode=self.args.testmode)
        z_fused = res["z_fused"]
        zeros = torch.zeros(a.shape[0], device=self.device)
        delta_z = self.side_car(
            [res["z_a"], res["z_v"]],
            [res["f_a"], res["f_v"]],
            [zeros, zeros],
        )
        z_final = z_fused + delta_z
        return (z_fused, z_final), (0.0, 0.0, 0.0, 0.0)

    # ─────────────────────────────────────────
    @torch.enable_grad()
    def _forward_and_adapt(self, a, v):
        self.t += 1
        B = a.shape[0]

        # ── 1. Backbone forward (frozen) ─────────────────────────
        with autocast():
            res = self.model.module.forward_eval_crrtta(a=a, v=v, mode=self.args.testmode)

        z_a    = res["z_a"]       # (B, C) logits
        z_v    = res["z_v"]
        z_fused = res["z_fused"]
        f_a    = res["f_a"]       # (B, D) GAP features
        f_v    = res["f_v"]

        p_a = z_a.softmax(-1)
        p_v = z_v.softmax(-1)

        # ── 2. Stochastic augmented views ────────────────────────
        aug_pa = self._get_aug_preds(a, v, modality="a")
        aug_pv = self._get_aug_preds(a, v, modality="v")

        # ── 3. Uncertainty decomposition ─────────────────────────
        _, _, _, U_a = compute_uncertainty(
            p_a, aug_pa, [p_v],
            self.alpha, self.beta, self.gamma_u
        )
        _, _, _, U_v = compute_uncertainty(
            p_v, aug_pv, [p_a],
            self.alpha, self.beta, self.gamma_u
        )

        # ── 4. Trust weights (temperature-scaled softmax) ────────
        w = torch.stack([-U_a / self.tau, -U_v / self.tau], dim=-1).softmax(-1)
        w_a, w_v = w[:, 0:1], w[:, 1:2]   # (B,1)

        # ── 5. EMA pseudo-target ──────────────────────────────────
        p_agg = w_a * p_a + w_v * p_v      # (B, C)
        if not self.p_ema_init:
            self.p_ema = p_agg.detach().clone()
            self.p_ema_init = True
        else:
            self.p_ema = self.mu * self.p_ema.detach() + (1 - self.mu) * p_agg.detach()

        # ── 6. Confidence gate ────────────────────────────────────
        min_unc = torch.min(torch.stack([U_a, U_v], -1), -1)[0]   # (B,)
        confident = min_unc < self.theta

        # ── 7. Side-Car forward ───────────────────────────────────
        delta_z = self.side_car(
            [z_a, z_v],
            [f_a, f_v],
            [U_a.detach(), U_v.detach()],
        )
        z_final = z_fused + delta_z
        p_final = z_final.softmax(-1)

        # ── 8. Loss ───────────────────────────────────────────────
        eps = 1e-9

        # L_align  –  KL(p_final || p_target) for confident samples
        if confident.any():
            p_target = sharpen(self.p_ema[confident])
            L_align = F.kl_div(
                p_final[confident].log().clamp(min=-1e9),
                p_target,
                reduction="batchmean",
            )
        else:
            # fall-back: standard entropy minimisation
            L_align = -(p_final * p_final.log().clamp(min=-1e9)).sum(-1).mean()

        # L_cons  –  consistency under perturbed inputs
        L_cons = self._consistency_loss(a, v, p_final.detach())

        # L_reg   –  residual regularisation
        L_reg = delta_z.pow(2).sum(-1).mean()

        # L_ent   –  curriculum entropy minimisation (warm-up)
        lambda1 = self.lambda_max * (1.0 - math.exp(-self.eta * self.t))
        L_ent = -(p_final * p_final.log().clamp(min=-1e9)).sum(-1).mean()

        L_total = (
            lambda1          * L_ent
            + self.lambda2   * L_cons
            + self.lambda3   * L_align
            + self.lambda4   * L_reg
        )

        # ── 9. Update only Side-Car params ───────────────────────
        self.optimizer.zero_grad()
        self.scaler.scale(L_total).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # ── 10. Inference pass after update ──────────────────────
        with torch.no_grad():
            with autocast():
                res2 = self.model.module.forward_eval_crrtta(
                    a=a, v=v, mode=self.args.testmode
                )
            dz2 = self.side_car(
                [res2["z_a"], res2["z_v"]],
                [res2["f_a"], res2["f_v"]],
                [U_a.detach(), U_v.detach()],
            )
            z_out = res2["z_fused"] + dz2

        losses = (L_ent.item(), L_cons.item(), L_align.item(), L_reg.item())
        return (z_fused, z_out), losses

    # ─────────────────────────────────────────
    @torch.no_grad()
    def _get_aug_preds(self, a, v, modality: str) -> list:
        """Return K augmented probability vectors for one modality."""
        preds = []
        with autocast():
            for _ in range(self.K_aug):
                if modality == "a":
                    a_in = a + self.noise_std * torch.randn_like(a)
                    v_in = v
                else:
                    a_in = a
                    v_in = v + self.noise_std * torch.randn_like(v)
                res = self.model.module.forward_eval_crrtta(
                    a=a_in, v=v_in, mode=self.args.testmode
                )
                key = "z_a" if modality == "a" else "z_v"
                preds.append(res[key].softmax(-1))
        return preds

    # ─────────────────────────────────────────
    def _consistency_loss(self, a, v, p_ref: torch.Tensor) -> torch.Tensor:
        """
        L_cons: average KL divergence between adapted output under K_aug
        perturbations and the reference (detached) output.
        """
        K = min(self.K_aug, 2)
        total = torch.tensor(0.0, device=self.device)

        with autocast():
            for _ in range(K):
                a_aug = a + self.noise_std * torch.randn_like(a)
                v_aug = v + self.noise_std * torch.randn_like(v)
                res_aug = self.model.module.forward_eval_crrtta(
                    a=a_aug, v=v_aug, mode=self.args.testmode
                )
                zeros = torch.zeros(a.shape[0], device=self.device)
                dz_aug = self.side_car(
                    [res_aug["z_a"], res_aug["z_v"]],
                    [res_aug["f_a"], res_aug["f_v"]],
                    [zeros, zeros],
                )
                p_aug = (res_aug["z_fused"] + dz_aug).softmax(-1)
                kl = F.kl_div(
                    p_aug.log().clamp(min=-1e9),
                    p_ref,
                    reduction="batchmean",
                )
                total = total + kl

        return total / K


# ─────────────────────────────────────────────
#  Model configuration helpers
# ─────────────────────────────────────────────

def configure_model(model: nn.Module) -> nn.Module:
    """Freeze the entire backbone. Side-Car is handled separately."""
    model.train()
    model.requires_grad_(False)
    return model


def collect_params(model):
    """
    Returns empty lists — the Side-Car lives inside CR_RTTA and its
    parameters are registered with the optimizer directly.
    """
    return [], []


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)
