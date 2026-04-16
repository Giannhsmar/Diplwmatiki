import os
import copy
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
import torchvision.utils as vutils

from efficient_net import build_efficientnet_lite
from utils import load_checkpoint, kaiming_init
from generator import Generator
from differentiable_augmentation import DiffAugment
from dataset import load_data

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s",
                    level=logging.INFO, datefmt="%I:%M:%S")

# ----------------------------- EfficientNet preproc -----------------------------
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def preprocess_for_effnet(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1] -> ImageNet norm
    x01 = (x + 1.0) * 0.5
    return (x01 - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)

# ----------------------------- Light, fringe-friendly losses -----------------------------

def sobel_map(x: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    kx = kx.repeat(x.size(1),1,1,1); ky = ky.repeat(x.size(1),1,1,1)
    gx = F.conv2d(x, kx, stride=1, padding=1, groups=x.size(1))
    gy = F.conv2d(x, ky, stride=1, padding=1, groups=x.size(1))
    g = torch.sqrt(gx*gx + gy*gy + 1e-8).mean(1, keepdim=True)
    return g

def edge_loss_simple(fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    gf, gr = sobel_map(fake), sobel_map(real)
    return (gf.mean() - gr.mean()).abs() + 0.5 * (gf.var(unbiased=False) - gr.var(unbiased=False)).abs()

# NB: bandpass_spectrum_loss returns a batch-aggregated discrepancy; we use it as a proxy
# for in-band energy/closeness.
def bandpass_spectrum_loss(fake: torch.Tensor, real: torch.Tensor, low=0.08, high=0.45) -> torch.Tensor:
    def _bp(x):
        xg = x.mean(1, keepdim=True)
        X = torch.fft.rfft2(xg, norm='ortho')
        B,C,H,Wc = X.shape
        yy = torch.fft.fftfreq(H, d=1.0, device=x.device).view(1,1,H,1).abs()
        xx = torch.fft.rfftfreq((Wc-1)*2, d=1.0, device=x.device).view(1,1,1,Wc).abs()
        r = torch.sqrt(xx*xx + yy*yy)
        mask = (r >= low) & (r <= high)
        P = (X.abs()**2) * mask
        P = P / (P.mean(dim=(2,3), keepdim=True) + 1e-8)
        return P.mean(0)
    return (_bp(fake) - _bp(real)).abs().mean()

# ----------------------------- Discriminator blocks -----------------------------

class DownBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = spectral_norm(nn.Conv2d(c_in, c_out, 4, 2, 1))
        # InstanceNorm breaks at 1x1; GroupNorm is robust
        num_groups = min(32, max(1, c_out // 8))
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=c_out)
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.act(x)
        return x

class MultiScaleDiscriminator(nn.Module):
    """Hinge-loss D with optional label projection."""
    def __init__(self, in_ch: int, depth: int, num_classes: int | None = None):
        super().__init__()
        widths = [128, 256, 512]
        layers = []
        last = in_ch
        for d in range(depth):
            out = widths[min(d, len(widths)-1)]
            layers.append(DownBlock(last, out))
            last = out
        self.model = nn.Sequential(*layers)
        self.head = spectral_norm(nn.Conv2d(last, 1, 3, 1, 1))
        self.embed = nn.Embedding(num_classes, last) if num_classes is not None else None
        self.optim = Adam(self.parameters(), lr=2e-4, betas=(0.0, 0.99))
    def forward(self, x, y: torch.Tensor | None = None):
        h = self.model(x)                     # [B,C,H,W]
        s = self.head(h).mean(dim=(2,3)).squeeze(1)  # [B]
        if (self.embed is not None) and (y is not None):
            pooled = h.mean(dim=(2,3))       # [B,C]
            proj = (pooled * self.embed(y)).sum(dim=1)
            s = s + proj
        return s

# ----------------------------- Cross-Scale Mixing -----------------------------

class CSM(nn.Module):
    def __init__(self, low_in: int, high_in: int | None, out_ch: int):
        super().__init__()
        self.low_proj  = nn.Conv2d(low_in, out_ch, kernel_size=1)
        self.high_proj = nn.Conv2d(high_in, out_ch, kernel_size=1) if high_in is not None else None
        self.refine    = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.apply(kaiming_init)
    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor | None = None):
        low = self.low_proj(low_res)
        if high_res is None:
            return self.refine(low)
        high = F.interpolate(high_res, scale_factor=2.0, mode="bilinear", align_corners=False)
        if self.high_proj is not None:
            high = self.high_proj(high)
        x = self.refine(low + high)
        return x

# ----------------------------- Main trainer -----------------------------

class ProjectedGAN:
    def __init__(self, args):
        self.img_size = args.image_size
        self.latent_dim = args.latent_dim
        self.epochs = args.epochs
        self.lr = args.lr
        self.log_every = args.log_every
        self.ckpt_path = args.checkpoint_path
        os.makedirs(self.ckpt_path, exist_ok=True)

        # Data (imagefolder DataLoader)
        base_loader = load_data(args.dataset_path, args.batch_size)
        base_ds = getattr(base_loader, "dataset", base_loader.dataset)
        labels_all = np.array(getattr(base_ds, "targets", []))
        self.base_ds = base_ds
        self.labels_all = labels_all
        self.batch_size = args.batch_size

        # Warmup class-1 only loader
        idx_cls1 = np.where(labels_all == 1)[0]
        self.loader_warm = DataLoader(Subset(base_ds, idx_cls1), batch_size=args.batch_size,
                                      shuffle=True, drop_last=True)

        # Models
        self.conditional = args.conditional
        self.num_classes = args.num_classes

        self.gen = Generator(im_size=args.image_size, nz=self.latent_dim)

        # Optional latent conditioning (no arch changes): z <- z + Emb[y]
        if self.conditional:
            self.z_embed = nn.Embedding(self.num_classes, self.latent_dim)
            with torch.no_grad():
                self.z_embed.weight.zero_()  # y=1 adds 0 during warmup; we'll unfreeze later
        else:
            self.z_embed = None

        # EfficientNet backbone (frozen)
        self.efficient_net = build_efficientnet_lite("efficientnet_lite1", 1000)
        self.efficient_net = nn.DataParallel(self.efficient_net)
        checkpoint = torch.load(args.checkpoint_efficient_net, map_location="cpu")
        load_checkpoint(self.efficient_net, checkpoint)
        self.efficient_net.eval()
        for p in self.efficient_net.parameters(): p.requires_grad = False

        # Feature sizes
        with torch.no_grad():
            dummy = torch.zeros(1,3,self.img_size,self.img_size)
            _, feats = self.efficient_net(preprocess_for_effnet(dummy))
            fs = [f.shape[1] for f in feats]  # [24,40,80,320]

        # CSM chain and Ds (expect CSM outputs)
        self.csms = nn.ModuleList([
            CSM(low_in=fs[3], high_in=None,   out_ch=fs[2]),  # 320->80
            CSM(low_in=fs[2], high_in=fs[2], out_ch=fs[1]),  # 80(+80)->40
            CSM(low_in=fs[1], high_in=fs[1], out_ch=fs[0]),  # 40(+40)->24
            CSM(low_in=fs[0], high_in=fs[0], out_ch=fs[0]),  # 24(+24)->24
        ])
        self.csm_optim = Adam(self.csms.parameters(), lr=2e-4, betas=(0.0, 0.99))

        def make_d(in_ch, depth):
            return MultiScaleDiscriminator(in_ch, depth, num_classes=self.num_classes if self.conditional else None)
        self.discs = nn.ModuleList([make_d(fs[2],3), make_d(fs[1],2), make_d(fs[0],1), make_d(fs[0],1)])

        # Optims
        gen_params = list(self.gen.parameters()) + (list(self.z_embed.parameters()) if self.z_embed is not None else [])
        self.gen_optim = Adam(gen_params, lr=self.lr, betas=(0.0, 0.99))
        for disc in self.discs:
            for pg in disc.optim.param_groups: pg['lr'] = 1e-4  # TTUR

        # DiffAug
        self.diff_aug = args.diff_aug
        self.DiffAug = DiffAugment(args.diffaug_policy)

        # Priors
        self.bp_low, self.bp_high = args.bp_low, args.bp_high
        self.bp_w, self.edge_w = args.bp_w, args.edge_w
        self.leak_margin, self.leak_w = args.leak_margin, args.leak_w

        # Curriculum
        self.warmup_epochs = args.warmup_epochs
        self.ramp_epochs = args.ramp_epochs

        # EMA
        self.gen_ema = copy.deepcopy(self.gen).eval()
        for p in self.gen_ema.parameters(): p.requires_grad = False
        self.ema_decay = 0.999

        self.save_all = args.save_all

    def csm_forward(self, features):
        features = features[::-1]  # [320,80,40,24]
        out = []
        for i, csm in enumerate(self.csms):
            if i == 0:
                d = csm(features[i])
            else:
                d = csm(features[i], d)
            out.append(d)
        return out  # [80,40,24,24]

    def _build_loader_for_epoch(self, epoch: int):
        if epoch < self.warmup_epochs:
            return self.loader_warm, 0.0
        # ramp class-0 proportion from 0 -> 0.5
        ramp_pos = min(1.0, (epoch - self.warmup_epochs + 1) / max(1, self.ramp_epochs))
        p0 = 0.5 * ramp_pos; p1 = 1.0 - p0
        counts = np.bincount(self.labels_all, minlength=2)
        w = np.zeros_like(self.labels_all, dtype=np.float32)
        w[self.labels_all == 0] = p0 / max(1, counts[0])
        w[self.labels_all == 1] = p1 / max(1, counts[1])
        sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)
        loader = DataLoader(self.base_ds, batch_size=self.batch_size, sampler=sampler, drop_last=True)
        return loader, p0

    def train(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Using device: {device}")
        self.gen.to(device); self.gen_ema.to(device)
        if self.z_embed is not None:            
            self.z_embed.to(device)

        for d in self.discs: d.to(device)
        for csm in self.csms: csm.to(device)
        self.efficient_net.to(device)
        self.vis_z = torch.randn(16, self.latent_dim, device=device)
        self.bp_ema = 1.0

        # Freeze z_embed during warmup (so y=1 acts as identity on z), then unfreeze
        if self.z_embed is not None:
            for p in self.z_embed.parameters(): p.requires_grad = False

        for epoch in range(self.epochs):
            loader, p0 = self._build_loader_for_epoch(epoch)
            if (self.z_embed is not None) and (epoch == self.warmup_epochs):
                for p in self.z_embed.parameters(): p.requires_grad = True
            logging.info(f"Starting epoch {epoch+1} (p0≈{p0:.2f})")

            for i, (real_imgs, labels) in enumerate(loader):
                real_imgs = real_imgs.to(device)
                y_real = labels.to(device).long()

                # -------------------- D update --------------------
                with torch.no_grad():
                    z = torch.randn(real_imgs.size(0), self.latent_dim, device=device)
                    if self.conditional:
                        y_fake = (torch.ones_like(y_real) if epoch < self.warmup_epochs else
                                  torch.randint(0, self.num_classes, (real_imgs.size(0),), device=device))
                        z = z + self.z_embed(y_fake)
                        fake = self.gen(z)
                    else:
                        y_fake = None
                        fake = self.gen(z)
                    fake = self.DiffAug.forward(fake) if self.diff_aug else fake
                    real_aug = self.DiffAug.forward(real_imgs) if self.diff_aug else real_imgs

                _, f_fake = self.efficient_net(preprocess_for_effnet(fake))
                _, f_real = self.efficient_net(preprocess_for_effnet(real_aug))
                f_fake = self.csm_forward(f_fake)
                f_real = self.csm_forward(f_real)

                for disc in self.discs: disc.optim.zero_grad(set_to_none=True)
                self.csm_optim.zero_grad(set_to_none=True)
                disc_losses = []
                disc_loss_total = 0.0
                do_r1 = (i % 32) == 0

                for fr, ff, disc in zip(f_real, f_fake, self.discs):
                    fr = fr.detach().requires_grad_(do_r1)
                    ff = ff.detach()
                    s_real = disc(fr, y_real if self.conditional else None)
                    s_fake = disc(ff, y_fake if self.conditional else None)
                    loss_real = F.relu(1 - s_real).mean()
                    loss_fake = F.relu(1 + s_fake).mean()
                    loss = loss_real + loss_fake
                    if do_r1:
                        grad = torch.autograd.grad(outputs=s_real.mean(), inputs=fr,
                                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
                        r1 = grad.pow(2).sum(dim=(1,2,3)).mean()
                        loss = loss + 0.5 * r1 * 0.5  # small weight
                    disc_loss_total = disc_loss_total + loss
                    disc_losses.append(loss.detach().item())
                disc_loss_total.backward()
                for disc in self.discs: disc.optim.step()
                self.csm_optim.step()

                # -------------------- G update(s) --------------------
                g_steps = 2 if (epoch < 10 or self.bp_ema > 0.7) else 1
                for gi in range(g_steps):
                    z = torch.randn(real_imgs.size(0), self.latent_dim, device=device)
                    if self.conditional:
                        y_fake = (torch.ones_like(y_real) if epoch < self.warmup_epochs else
                                  torch.randint(0, self.num_classes, (real_imgs.size(0),), device=device))
                        z = z + self.z_embed(y_fake)
                        gen = self.gen(z)
                    else:
                        y_fake = None
                        gen = self.gen(z)
                    gen = self.DiffAug.forward(gen) if self.diff_aug else gen

                    _, f_fake_g = self.efficient_net(preprocess_for_effnet(gen))
                    f_fake_g = self.csm_forward(f_fake_g)

                    gen_loss = 0.0
                    for ff, disc in zip(f_fake_g, self.discs):
                        s_fake = disc(ff, y_fake if self.conditional else None)
                        gen_loss = gen_loss + (-s_fake.mean())

                    # BP/EDGE only on class-1 fakes
                    if self.conditional:
                        m1 = (y_fake == 1)
                    else:
                        m1 = torch.ones(gen.size(0), dtype=torch.bool, device=device)
                    if m1.any():
                        real_ref = real_imgs[m1] if (self.conditional and (y_real==1).any()) else real_imgs
                        bp = bandpass_spectrum_loss(gen[m1], real_ref, low=self.bp_low, high=self.bp_high)
                        ed = edge_loss_simple(gen[m1], real_ref)

                        # leakage margin: want E0 > E1 by margin
                        if self.conditional and (self.num_classes == 2):
                            m0 = (y_fake == 0)
                            if m0.any():
                                E1 = bandpass_spectrum_loss(gen[m1], real_ref, low=self.bp_low, high=self.bp_high)
                                E0 = bandpass_spectrum_loss(gen[m0], real_ref, low=self.bp_low, high=self.bp_high)
                                leak_hinge = F.relu(self.leak_margin - (E0 - E1))
                            else:
                                leak_hinge = 0.0
                        else:
                            leak_hinge = 0.0

                        self.bp_ema = 0.95 * self.bp_ema + 0.05 * float(bp.detach().item())
                        bp_w_eff = self.bp_w if self.bp_ema > 0.6 else max(self.bp_w, 0.0025)
                        gen_loss = gen_loss + bp_w_eff * bp + self.edge_w * ed
                        if isinstance(leak_hinge, torch.Tensor):
                            gen_loss = gen_loss + self.leak_w * leak_hinge

                    self.gen_optim.zero_grad(set_to_none=True)
                    gen_loss.backward(retain_graph=(gi < g_steps - 1))
                    self.gen_optim.step()

                    with torch.no_grad():
                        for p, q in zip(self.gen.parameters(), self.gen_ema.parameters()):
                            q.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

                # -------------------- Logging --------------------
                if i % self.log_every == 0:                 # ← log gate (level 1)
                    path = os.path.join(self.ckpt_path, str(epoch))
                    os.makedirs(path, exist_ok=True)
                    with torch.no_grad():                   # ← no_grad (level 2)
                        vutils.save_image(gen.add(1).mul(0.5).clamp_(0,1), os.path.join(path, f"{epoch}_{i}.jpg"), nrow=4)
                        cur = self.gen(self.vis_z if not self.conditional else (self.vis_z + self.z_embed(torch.ones(16, dtype=torch.long, device=device))))
                        vutils.save_image(cur.add(1).mul(0.5).clamp_(0,1), os.path.join(path, f"{epoch}_{i}_cur.jpg"), nrow=4)
                        ema = self.gen_ema(self.vis_z if not self.conditional else (self.vis_z + self.z_embed(torch.ones(16, dtype=torch.long, device=device))))
                        vutils.save_image(ema.add(1).mul(0.5).clamp_(0,1), os.path.join(path, f"{epoch}_{i}_ema.jpg"), nrow=4)

                        if self.conditional:                 # ← class-specific samples (level 3)
                            y0 = torch.zeros(16, dtype=torch.long, device=device)
                            y1 = torch.ones(16,  dtype=torch.long, device=device)

                            cur0 = self.gen(self.vis_z + self.z_embed(y0))
                            cur1 = self.gen(self.vis_z + self.z_embed(y1))
                            vutils.save_image(cur0.add(1).mul(0.5).clamp_(0,1),
                                                os.path.join(path, f"{epoch}_{i}_cur_y0.jpg"), nrow=4)
                            vutils.save_image(cur1.add(1).mul(0.5).clamp_(0,1),
                                                os.path.join(path, f"{epoch}_{i}_cur_y1.jpg"), nrow=4)

                            ema0 = self.gen_ema(self.vis_z + self.z_embed(y0))
                            ema1 = self.gen_ema(self.vis_z + self.z_embed(y1))
                            vutils.save_image(ema0.add(1).mul(0.5).clamp_(0,1),
                                                os.path.join(path, f"{epoch}_{i}_ema_y0.jpg"), nrow=4)
                            vutils.save_image(ema1.add(1).mul(0.5).clamp_(0,1),
                                                os.path.join(path, f"{epoch}_{i}_ema_y1.jpg"), nrow=4)

                    logging.info(f"Iter {i}: G={gen_loss.item():.4f} (BP_EMA {self.bp_ema:.3f}, p0≈{p0:.2f}) D={ [round(x,4) for x in disc_losses] }")

                    torch.save(self.gen.state_dict(), os.path.join(path, "Generator"))

                    # NEW: save the class embedding so you can condition at inference
                    if self.conditional and (self.z_embed is not None):
                        z_tensor = self.z_embed.weight.detach().clone().cpu()
                        torch.save(
                            {
                                "z_embed": z_tensor,                 # shape: (num_classes, latent_dim)
                                "num_classes": self.num_classes,
                                "latent_dim": self.latent_dim,
                            },
                            os.path.join(path, "z_embed.pt")
                        )

                    if self.save_all:
                        for j,disc in enumerate(self.discs):
                            torch.save(disc.state_dict(), os.path.join(path, f"Discriminator_{j}"))
                            torch.save(self.csms[j].state_dict(), os.path.join(path, f"CSM_{j}"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conditional ProjectedGAN')
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--checkpoint-efficient-net', type=str, required=True)
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints_insar_base')
    parser.add_argument('--save-all', type=bool, default=True)
    parser.add_argument('--diff-aug', type=bool, default=True)
    parser.add_argument('--diffaug-policy', type=str, default='translation')

    # Conditioning / curriculum
    parser.add_argument('--conditional', type=bool, default=True)
    parser.add_argument('--num-classes', type=int, default=2)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--ramp-epochs', type=int, default=5)

    # Fringe priors & leakage control
    parser.add_argument('--bp-low', type=float, default=0.12)
    parser.add_argument('--bp-high', type=float, default=0.30)
    parser.add_argument('--bp-w', type=float, default=0.0015)
    parser.add_argument('--edge-w', type=float, default=0.006)
    parser.add_argument('--leak-margin', type=float, default=0.10)
    parser.add_argument('--leak-w', type=float, default=0.5)

    args = parser.parse_args()
    trainer = ProjectedGAN(args)
    trainer.train()
