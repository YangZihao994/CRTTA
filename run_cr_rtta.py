import argparse
import os
os.environ['MPLCONFIGDIR'] = './plt/'
import sys
import warnings

import numpy as np
import torch
from tqdm import tqdm

import dataloader as dataloader
import models
from utilities import accuracy, seed_everything
from TTA import CR_RTTA


# ─────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# ── dataset ──
parser.add_argument('--dataset',      type=str, default='vggsound',
                    choices=['vggsound', 'ks50'])
parser.add_argument('--json-root',    type=str,
                    default='json_csv_files/vgg')
parser.add_argument('--label-csv',    type=str,
                    default='json_csv_files/class_labels_indices_vgg.csv')
parser.add_argument('--n_class',      type=int, default=309)
parser.add_argument('--dataset_mean', type=float, default=-5.081)
parser.add_argument('--dataset_std',  type=float, default=4.4849)
parser.add_argument('--target_length',type=int,   default=1024)

# ── model ──
parser.add_argument('--model',        type=str, default='cav-mae-ft')
parser.add_argument('--pretrain_path',type=str, default='pretrained_model/vgg_65.5.pth')
parser.add_argument('--testmode',     type=str, default='multimodal')
parser.add_argument('--feat-dim',     type=int, default=768,
                    help='backbone embedding dim (768 for CAV-MAE)')
parser.add_argument('--proj-dim',     type=int, default=128,
                    help='side-car projection dim')

# ── training ──
parser.add_argument('--lr',   default=1e-4, type=float)
parser.add_argument('--optim',default='adam', choices=['adam', 'sgd'])
parser.add_argument('-b', '--batch-size', default=64,   type=int)
parser.add_argument('-w', '--num-workers',default=32,   type=int)
parser.add_argument('--gpu',  default='0', type=str)

# ── corruption ──
parser.add_argument('--corruption-modality', type=str,
                    default='video', choices=['video', 'audio', 'none'])
parser.add_argument('--severity-start', type=int, default=5)
parser.add_argument('--severity-end',   type=int, default=5)

# ── tta / uncertainty ──
parser.add_argument('--tta-method',   type=str, default='CR_RTTA',
                    choices=['CR_RTTA', 'None'])
parser.add_argument('--unc-alpha',    type=float, default=1.0)
parser.add_argument('--unc-beta',     type=float, default=1.0)
parser.add_argument('--unc-gamma',    type=float, default=1.0)
parser.add_argument('--tau',          type=float, default=1.0)
parser.add_argument('--mu',           type=float, default=0.9,
                    help='EMA decay for pseudo-target')
parser.add_argument('--theta',        type=float, default=0.5,
                    help='confidence gate threshold')
parser.add_argument('--K-aug',        type=int,   default=4,
                    help='number of stochastic augmentation views')
parser.add_argument('--aug-noise',    type=float, default=0.01,
                    help='Gaussian noise std for lightweight augmentation')
parser.add_argument('--lambda-max',   type=float, default=1.0)
parser.add_argument('--lambda2',      type=float, default=1.0)
parser.add_argument('--lambda3',      type=float, default=1.0)
parser.add_argument('--lambda4',      type=float, default=0.01)
parser.add_argument('--eta',          type=float, default=0.01,
                    help='warm-up growth rate for entropy weight')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# ── adjust n_class ────────────────────────────────────────────────
if args.dataset == 'vggsound':
    args.n_class = 309
elif args.dataset == 'ks50':
    args.n_class = 50
# expose under both names so SideCar can read it
args.feat_dim = args.feat_dim
args.proj_dim = args.proj_dim

print(args)

# ─────────────────────────────────────────────────────────────────
#  Corruption lists
# ─────────────────────────────────────────────────────────────────

if args.corruption_modality == 'video':
    corruption_list = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]
elif args.corruption_modality == 'audio':
    corruption_list = [
        'gaussian_noise', 'traffic', 'crowd', 'rain', 'thunder', 'wind',
    ]
else:
    corruption_list = ['clean']
    args.severity_start = args.severity_end = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────

for corruption in corruption_list:
    for severity in range(args.severity_start, args.severity_end + 1):
        epoch_accs = []

        if args.corruption_modality == 'none':
            data_val = os.path.join(
                args.json_root, corruption, f'severity_{severity}.json')
        else:
            data_val = os.path.join(
                args.json_root, args.corruption_modality,
                corruption, f'severity_{severity}.json')

        print(f'\n===> {data_val}')

        for itr in range(1, 6):
            seed = int(str(itr) * 3)
            seed_everything(seed=seed)
            print(f'### Seed={seed}, Round {itr} ###')

            # ── dataloader ───────────────────────────────────────
            im_res = 224
            val_conf = {
                'num_mel_bins': 128, 'target_length': args.target_length,
                'freqm': 0, 'timem': 0, 'mixup': 0,
                'dataset': args.dataset, 'mode': 'eval',
                'mean': args.dataset_mean, 'std': args.dataset_std,
                'noise': False, 'im_res': im_res,
            }
            tta_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(data_val,
                                           label_csv=args.label_csv,
                                           audio_conf=val_conf),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )

            # ── backbone ─────────────────────────────────────────
            if args.model == 'cav-mae-ft':
                va_model = models.CAVMAEFT(
                    label_dim=args.n_class,
                    modality_specific_depth=11,
                )
            else:
                raise ValueError(f'Unknown model: {args.model}')

            if args.pretrain_path != 'None':
                mdl_weight = torch.load(args.pretrain_path, map_location='cpu')
                if not isinstance(va_model, torch.nn.DataParallel):
                    va_model = torch.nn.DataParallel(va_model)
                miss, unexp = va_model.load_state_dict(mdl_weight, strict=False)
                print(f'Loaded weights from {args.pretrain_path}')
                print(f'  missing={miss}\n  unexpected={unexp}')
            else:
                warnings.warn('No pretrained weights specified.')
                if not isinstance(va_model, torch.nn.DataParallel):
                    va_model = torch.nn.DataParallel(va_model)

            va_model.to(device)

            # ── configure (freeze backbone) ───────────────────────
            va_model = CR_RTTA.configure_model(va_model)

            adapt_flag = args.tta_method != 'None'

            if args.tta_method in ('CR_RTTA', 'None'):
                # Side-Car lives inside the CR_RTTA module
                cr_model = CR_RTTA.CR_RTTA(va_model, None, device, args)
                cr_model.to(device)

                # Optimise only side-car parameters
                sidecar_params = list(cr_model.side_car.parameters())
                total_p     = sum(p.numel() for p in va_model.parameters())
                trainable_p = sum(p.numel() for p in sidecar_params)
                print(f'Backbone params : {total_p / 1e6:.3f} M  (frozen)')
                print(f'Side-Car params : {trainable_p / 1e6:.3f} M  (trainable)')

                if args.optim == 'adam':
                    optimizer = torch.optim.Adam(
                        sidecar_params, lr=args.lr,
                        betas=(0.9, 0.999), weight_decay=0.0,
                    )
                else:
                    optimizer = torch.optim.SGD(
                        sidecar_params, lr=args.lr, momentum=0.9,
                    )
                cr_model.optimizer = optimizer

                # ── eval loop ────────────────────────────────────
                cr_model.eval()      # keep BN/dropout in eval mode
                with torch.no_grad():
                    pass             # no outer no_grad — adapt needs grads

                batch_accs = []
                data_bar = tqdm(tta_loader)

                for i, (a_in, v_in, labels) in enumerate(data_bar):
                    a_in = a_in.to(device)
                    v_in = v_in.to(device)

                    (z_pre, z_post), losses = cr_model(
                        (a_in, v_in), adapt_flag=adapt_flag
                    )

                    # evaluate on the post-adaptation output
                    acc = accuracy(z_post, labels, topk=(1,))[0].item()
                    batch_accs.append(round(acc, 2))

                    l_ent, l_cons, l_align, l_reg = losses
                    data_bar.set_description(
                        f'B#{i} Lent={l_ent:.4f} Lcons={l_cons:.4f} '
                        f'Lalign={l_align:.4f} Lreg={l_reg:.4f} acc={acc:.2f}'
                    )

                epoch_acc = round(sum(batch_accs) / len(batch_accs), 2)
                epoch_accs.append(epoch_acc)
                print(f'Epoch acc: {epoch_acc}')

        mean_acc = np.round(np.mean(epoch_accs), 2)
        std_acc  = np.round(np.std(epoch_accs),  2)
        print(f'===> {corruption}-sev{severity}  mean={mean_acc}  std={std_acc}')
