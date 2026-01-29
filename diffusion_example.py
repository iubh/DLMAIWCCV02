import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

# -------------------------------
# 1. Diffusion Process
# -------------------------------
class Diffusion:
    def __init__(self, T=200, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.T = T
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise):
        a_bar = self.alpha_bar[t][:, None, None, None]
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise

    def p_sample(self, xt, t, eps_pred):
        beta = self.betas[t][:, None, None, None]
        alpha = self.alphas[t][:, None, None, None]
        a_bar = self.alpha_bar[t][:, None, None, None]

        mean = (1 / torch.sqrt(alpha)) * (xt - beta / torch.sqrt(1 - a_bar) * eps_pred)

        if t.min() == 0:
            return mean

        noise = torch.randn_like(xt)
        var = beta * (1 - self.alpha_bar[t-1][:,None,None,None]) / (1 - a_bar)
        return mean + torch.sqrt(var) * noise

# -------------------------------
# 2. Conditional UNet
# -------------------------------
class DepthUNet(nn.Module):
    def __init__(self, ch=64):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(1, ch),
            nn.ReLU(),
            nn.Linear(ch, ch)
        )

        self.enc1 = nn.Conv2d(4, ch, 3, padding=1)
        self.enc2 = nn.Conv2d(ch, ch*2, 3, stride=2, padding=1)
        self.enc3 = nn.Conv2d(ch*2, ch*4, 3, stride=2, padding=1)

        self.dec2 = nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2, padding=1)
        self.dec1 = nn.ConvTranspose2d(ch*2, ch, 4, stride=2, padding=1)

        self.out = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, depth_noisy, rgb, t):
        t = t.float().view(-1, 1) / 1000
        t_emb = self.time_mlp(t)[:, :, None, None]

        x = torch.cat([depth_noisy, rgb], dim=1)
        x = self.enc1(x) + t_emb
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.dec2(x)
        x = self.dec1(x)
        return self.out(x)

# -------------------------------
# 3. Dummy Dataset
# -------------------------------
class DummyDepthDataset(Dataset):
    def __init__(self, n=500):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rgb = torch.rand(3, 64, 64)
        depth = torch.rand(1, 64, 64)
        return rgb, depth

# -------------------------------
# 4. Training Step
# -------------------------------
def train_step(model, diffusion, rgb, depth, opt):
    B = depth.size(0)
    t = torch.randint(0, diffusion.T, (B,), device=depth.device)
    noise = torch.randn_like(depth)

    x_t = diffusion.q_sample(depth, t, noise)
    pred = model(x_t, rgb, t)

    loss = nn.functional.mse_loss(pred, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()

# -------------------------------
# 5. Sampling
# -------------------------------
@torch.no_grad()
def sample(model, diffusion, rgb):
    B, _, H, W = rgb.shape
    xt = torch.randn(B, 1, H, W, device=rgb.device)

    for t in reversed(range(diffusion.T)):
        t_batch = torch.full((B,), t, device=rgb.device)
        eps = model(xt, rgb, t_batch)
        xt = diffusion.p_sample(xt, t_batch, eps)

    return xt

# -------------------------------
# 6. Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DepthUNet().to(device)
    diffusion = Diffusion(device=device)

    if args.mode == "train":
        loader = DataLoader(DummyDepthDataset(), batch_size=8, shuffle=True)
        opt = optim.Adam(model.parameters(), lr=1e-4)

        print("Training...")
        for epoch in range(args.epochs):
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for rgb, depth in pbar:
                rgb, depth = rgb.to(device), depth.to(device)
                loss = train_step(model, diffusion, rgb, depth, opt)
                pbar.set_postfix({"loss": f"{loss:.4f}"})

        torch.save(model.state_dict(), "depth_diffusion.pth")
        print("Model saved as depth_diffusion.pth")

    else:  # infer
        model.load_state_dict(torch.load("depth_diffusion.pth", map_location=device))
        model.eval()

        rgb = torch.rand(1, 3, 64, 64).to(device)
        depth = sample(model, diffusion, rgb)

        print("Inference done.")
        print("Predicted depth shape:", depth.shape)

if __name__ == "__main__":
    main()
