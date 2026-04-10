from pathlib import Path
from vae_demo import model
from vae_demo.spiral import SpiralDataset, Spiral
from vae_demo.config import VAEConfig
import matplotlib.pyplot as plt
import torch
import numpy as np


rc = Path().home() / ".config/matplotlib/matplotlibrc"
if rc.exists():
    plt.style.use(rc)

save_dir = Path("results/basic_run")
save_dir.mkdir(parents=True, exist_ok=True)


sigma = 0.2
cfg = VAEConfig(
    decoder_sigma=sigma,
    spread=sigma,
    omega=2.8 * np.pi,
    rdot=2.5,
    epochs=20000,
    t_steps=200,
    t_start=0.01,
)
cfg.save(save_dir / "config.json")
encoder, decoder, kl_loss_hist, re_loss_hist, loss_hist, dset = model.train(cfg)
torch.save(encoder.state_dict(), save_dir / "encoder_state_dict.pt")
torch.save(decoder.state_dict(), save_dir / "decoder_state_dict.pt")

spiral = Spiral(omega=dset.omega, rdot=dset.rdot, spread=dset.spread)
x_exact, y_exact = spiral.exact(dset.ts)

with torch.no_grad():
    mu, logsigma = encoder(dset.data)
    z = model.reparam(mu, logsigma)
    output = decoder(z)

    exact_data = torch.from_numpy(np.concatenate(np.array([x_exact, y_exact]), axis=1))
    mu2, logsigma2 = encoder(exact_data)
    z2 = model.reparam(mu2, logsigma2)
    output2 = decoder(z2)

with torch.no_grad():
    output3 = decoder(torch.randn((150, 1)))
    output3 = output3 + sigma * torch.randn_like(output3)

plt.figure()
plt.plot(loss_hist)
plt.yscale("log")

fig, ax = plt.subplots(1, 3, figsize=(8, 4.5))
ax[0].plot(dset.ts.reshape(-1), mu.numpy(), ".")
ax[0].set_xlabel("t")
ax[0].set_ylabel(r"$\mu_\phi(x,y)$")
ax[0].set_title("Latent Space")
ax[0].set_xlim([0, 1.1])

ax[1].plot(x_exact, y_exact, label="exact")
ax[1].plot(dset[:, 0], dset[:, 1], ".", color="black", alpha=0.5, label="data")
ax[1].plot(
    output[:, 0], output[:, 1], ".", color="red", alpha=0.5, label="reconstruction"
)
ax[1].legend()
ax[1].set_title("Reconstruction")
ax[1].set_xticks([])
ax[1].set_yticks([])

ax[2].plot(x_exact, y_exact, label="reference", color="grey", alpha=0.5, linewidth=1)
ax[2].plot(output3[:, 0], output3[:, 1], ".", color="red", alpha=0.5, label="generated")
ax[2].set_title("Generated Samples")
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].legend()

fig.savefig(save_dir / "basic_run.svg")
plt.show()
