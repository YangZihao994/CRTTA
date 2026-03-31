import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
from torch.cuda.amp import autocast, GradScaler


# =====================================================================
#  数值安全的辅助函数 (Float32 护甲)
# =====================================================================

def sharpen(p, T=0.25):
    p = p.float().clamp(min=1e-9)
    return F.softmax(torch.log(p) / T, dim=-1)

def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.float()
    q = q.float()
    m = (0.5 * (p + q)).clamp(min=1e-9)
    
    kl_pm = F.kl_div(m.log(), p, reduction="none").sum(-1)
    kl_qm = F.kl_div(m.log(), q, reduction="none").sum(-1)
    
    js = 0.5 * (kl_pm + kl_qm).clamp(min=0.0)
    return js / math.log(2.0)

def compute_uncertainty(p_m, aug_pm_list, other_pm_list, alpha, beta, gamma):
    p_m = p_m.float()
    B, C = p_m.shape
    
    H_m = -(p_m * p_m.clamp(min=1e-9).log()).sum(-1) / math.log(C)
    
    C_m = torch.zeros(B, device=p_m.device)
    if len(aug_pm_list) > 0:
        for p_aug in aug_pm_list:
            C_m += js_divergence(p_m, p_aug)
        C_m = 1.0 - (C_m / len(aug_pm_list))
    else:
        C_m = torch.ones(B, device=p_m.device)
        
    A_m = torch.zeros(B, device=p_m.device)
    if len(other_pm_list) > 0:
        for p_other in other_pm_list:
            A_m += js_divergence(p_m, p_other)
        A_m = 1.0 - (A_m / len(other_pm_list))
    else:
        A_m = torch.ones(B, device=p_m.device)
        
    U_m = alpha * H_m + beta * (1 - C_m) + gamma * (1 - A_m)
    return H_m, C_m, A_m, U_m


# =====================================================================
#  Feature-Aware Residual Side-Car
# =====================================================================

class SideCar(nn.Module):
    def __init__(
        self,
        n_class: int,
        feat_dim: int = 768,
        proj_dim: int = 64,  # 📉 降低降维后的维度 (原128)
        n_modalities: int = 2,
    ):
        super().__init__()
        # 使用更简单的降维
        self.W_proj = nn.Linear(feat_dim, proj_dim)

        in_dim = n_class * n_modalities + proj_dim * n_modalities + n_modalities
        hidden = 64 # 📉 极度缩减隐藏层 (原512)

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden), # LayerNorm 前置，压制特征数值
            nn.ReLU(),
            nn.Linear(hidden, n_class),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # 零初始化
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, logits_list, feat_list, u_list):
        proj = [F.relu(self.W_proj(f)) for f in feat_list]
        u_cols = [u.unsqueeze(-1) for u in u_list]
        x = torch.cat(logits_list + proj + u_cols, dim=-1)
        return self.mlp(x)


# =====================================================================
#  CR-RTTA (Main Module)
# =====================================================================

class CR_RTTA(nn.Module):
    def __init__(self, model, optimizer, device, args, steps: int = 1):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.args = args
        self.device = device
        self.scaler = GradScaler()

        self.alpha   = getattr(args, "unc_alpha",   1.0)
        self.beta    = getattr(args, "unc_beta",    1.0)
        self.gamma_u = getattr(args, "unc_gamma",   1.0)
        self.tau     = getattr(args, "tau",         1.0)
        self.theta   = getattr(args, "theta",       0.5)
        self.K_aug   = getattr(args, "K_aug",       4)

        self.lambda_max = getattr(args, "lambda_max", 1.0)
        self.lambda2    = getattr(args, "lambda2",    1.0)
        self.lambda3    = getattr(args, "lambda3",    1.0)
        self.lambda4    = getattr(args, "lambda4",    0.01)
        self.eta        = getattr(args, "eta",        0.05)

        self.t = 0 

        n_class  = args.n_class
        feat_dim = getattr(args, "feat_dim",  768)
        proj_dim = getattr(args, "proj_dim",  128)
        self.side_car = SideCar(n_class, feat_dim, proj_dim, n_modalities=2).to(device)

    def forward(self, x, adapt_flag: bool):
        a_input, v_input = x

        for _ in range(self.steps):
            if adapt_flag:
                out, loss = self._forward_and_adapt(a_input, v_input)
            else:
                out, loss = self._forward_only(a_input, v_input)

        return out, loss

    @torch.no_grad()
    def _forward_only(self, a, v):
        with autocast():
            res = self.model.module.forward_eval_crrtta(a=a, v=v, mode=self.args.testmode)
        z_fused = res["z_fused"]
        return (z_fused, z_fused), (0.0, 0.0, 0.0, 0.0)

    @torch.enable_grad()
    def _forward_and_adapt(self, a, v):
        self.t += 1
        B = a.shape[0]

        with autocast():
            res = self.model.module.forward_eval_crrtta(a=a, v=v, mode=self.args.testmode)

        z_a     = res["z_a"]
        z_v     = res["z_v"]
        z_fused = res["z_fused"]
        f_a     = res["f_a"]
        f_v     = res["f_v"]

        p_a = z_a.softmax(-1)
        p_v = z_v.softmax(-1)

        aug_pa = self._get_aug_preds(a, v, modality="a")
        aug_pv = self._get_aug_preds(a, v, modality="v")

        _, _, _, U_a = compute_uncertainty(
            p_a, aug_pa, [p_v],
            self.alpha, self.beta, self.gamma_u
        )
        _, _, _, U_v = compute_uncertainty(
            p_v, aug_pv, [p_a],
            self.alpha, self.beta, self.gamma_u
        )

        w = torch.stack([-U_a / self.tau, -U_v / self.tau], dim=-1).softmax(-1)
        w_a, w_v = w[:, 0:1], w[:, 1:2]

        p_target_base = (w_a * p_a + w_v * p_v).detach()

        min_unc = torch.min(torch.stack([U_a, U_v], -1), -1)[0]
        confident = min_unc < self.theta

        delta_z = self.side_car(
            [z_a, z_v],
            [f_a, f_v],
            [U_a.detach(), U_v.detach()],
        )
        
        # 物理防爆钳制
        delta_z = torch.clamp(delta_z, min=-5.0, max=5.0)

        z_final = z_fused + delta_z
        p_final = z_final.softmax(-1)
        log_p_final = F.log_softmax(z_final, dim=-1)

        if confident.any():
            p_target = sharpen(p_target_base[confident])
            L_align = F.kl_div(
                log_p_final[confident].float(), 
                p_target.float(),
                reduction="batchmean",
            )
        else:
            L_align = -(p_final.float() * log_p_final.float()).sum(-1).mean()

        # 特征空间一致性损失调用
        L_cons = self._consistency_loss(
            z_a, z_v, f_a, f_v, z_fused, 
            p_final.detach(), U_a.detach(), U_v.detach()
        )
        
        L_ent = -(p_final.float() * log_p_final.float()).sum(-1).mean()
        L_reg = (delta_z.float() ** 2).sum(-1).mean()

        # 动态 Curriculum 预热
        lambda1_t = self.lambda_max * (1.0 - math.exp(-self.eta * self.t))

        L_total = (
            lambda1_t    * L_ent
            + self.lambda2 * L_cons
            + self.lambda3 * L_align
            + self.lambda4 * L_reg
        )

        self.optimizer.zero_grad()
        self.scaler.scale(L_total).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.side_car.parameters(), max_norm=5.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

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
            dz2 = torch.clamp(dz2, min=-5.0, max=5.0)
            z_out = res2["z_fused"] + dz2

        losses = (L_ent.item(), L_cons.item(), L_align.item(), L_reg.item())
        return (z_fused, z_out), losses

    @torch.no_grad()
    def _get_aug_preds(self, a, v, modality: str) -> list:
        preds = []
        with autocast():
            for _ in range(self.K_aug):
                if modality == "a":
                    a_in = F.dropout(a, p=0.2, training=True)
                    v_in = v
                else:
                    a_in = a
                    v_in = F.dropout(v, p=0.2, training=True)
                res = self.model.module.forward_eval_crrtta(
                    a=a_in, v=v_in, mode=self.args.testmode
                )
                key = "z_a" if modality == "a" else "z_v"
                preds.append(res[key].softmax(-1))
        return preds

    def _consistency_loss(self, z_a, z_v, f_a, f_v, z_fused, p_ref: torch.Tensor, U_a: torch.Tensor, U_v: torch.Tensor) -> torch.Tensor:
        K = min(self.K_aug, 2)
        total = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        T_cons = 2.0 
        p_ref_smooth = F.softmax(p_ref.float().clamp(min=1e-9).log() / T_cons, dim=-1)

        with torch.cuda.amp.autocast():
            for _ in range(K):
                f_a_aug = F.dropout(f_a, p=0.1, training=True) + 0.05 * torch.randn_like(f_a)
                f_v_aug = F.dropout(f_v, p=0.1, training=True) + 0.05 * torch.randn_like(f_v)
                
                z_a_aug = F.dropout(z_a, p=0.1, training=True)
                z_v_aug = F.dropout(z_v, p=0.1, training=True)

                dz_aug = self.side_car(
                    [z_a_aug, z_v_aug],
                    [f_a_aug, f_v_aug],
                    [U_a, U_v],
                )
                dz_aug = torch.clamp(dz_aug, min=-5.0, max=5.0)
                
                z_fused_aug = F.dropout(z_fused, p=0.1, training=True)
                z_aug = z_fused_aug + dz_aug
                
                log_p_aug_smooth = F.log_softmax(z_aug.float() / T_cons, dim=-1)
                kl = F.kl_div(
                    log_p_aug_smooth,
                    p_ref_smooth,
                    reduction="batchmean",
                ) * (T_cons ** 2)
                
                total = total + kl

        return total / K


# =====================================================================
#  Model Configuration Helpers
# =====================================================================

def configure_model(model: nn.Module) -> nn.Module:
    model.train()
    model.requires_grad_(False)
    return model

def collect_params(model):
    return [], []

def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)