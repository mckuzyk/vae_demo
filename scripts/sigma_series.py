from pathlib import Path
from vae_demo.config import VAEConfig
from vae_demo.model import train, reparam
import matplotlib.pyplot as plt
import torch
from vae_demo.spiral import Spiral
import numpy as np

rc = Path().home() / ".config/matplotlib/matplotlibrc"
if rc.exists():
    plt.style.use(rc)

save_dir = Path("results/sigma_series")
save_dir.mkdir(parents=True, exist_ok=True)

spread = 0.2
sigmas = [0.0001, 0.2, 2.0]
data = {}
for i, sigma in enumerate(sigmas):
    cur_data = {}
    cfg = VAEConfig(
        spread=spread,
        decoder_sigma=sigma,
        omega=2.8 * np.pi,
        rdot=2.5,
        epochs=15000,
        t_steps=200,
        t_start=0.05,
        encoder_hidden_layers=2,
        decoder_hidden_layers=2,
    )
    cfg.save(save_dir / "config_{i}.json")
    encoder, decoder, kl_loss_hist, re_loss_hist, loss_hist, dset = train(cfg)
    torch.save(encoder.state_dict(), save_dir / "encoder_state_dict_{i}.pt")
    torch.save(decoder.state_dict(), save_dir / "decoder_state_dict_{i}.pt")
    input = dset.data
    with torch.no_grad():
        mu, logsigma = encoder(input)
        z = reparam(mu, logsigma)
        output = decoder(z)

    cur_data = {
        "encoder": encoder,
        "decoder": decoder,
        "kl": kl_loss_hist,
        "re": re_loss_hist,
        "loss": loss_hist,
        "dset": dset,
        "output": output,
        "mu": mu.numpy(),
        "sigma_phi": np.exp(logsigma.numpy()),
    }
    data[sigma] = cur_data


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 4.5), sharey=True)
for i, sigma in enumerate(data.keys()):
    _data = data[sigma]
    dset = _data["dset"]
    ts = _data["dset"].ts
    spiral = Spiral(omega=dset.omega, rdot=dset.rdot, spread=dset.spread)
    exact_x, exact_y = spiral.exact(ts)
    ax[i].plot(exact_x, exact_y)
    ax[i].plot(dset[:, 0], dset[:, 1], ".", color="black", alpha=0.5)
    ax[i].plot(
        _data["output"][:, 0], _data["output"][:, 1], ".", color="red", alpha=0.5
    )
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(r"$\sigma / \sigma_{data} = $" + f"{sigma / spread}")


fig2, ax2 = plt.subplots(figsize=(8, 2.5))
ax2.plot(
    data[sigmas[-1]]["dset"].ts.numpy().reshape(-1),
    data[sigmas[-1]]["mu"].reshape(-1),
    ".",
    label=r"$\mu_\phi(x,y)^2$",
)
ax2.plot(
    data[sigmas[-1]]["dset"].ts.numpy().reshape(-1),
    data[sigmas[-1]]["sigma_phi"].reshape(-1),
    ".",
    label=r"$\sigma_\phi(x,y)^2$",
)
ax2.legend()
ax2.set_title("Recognition Model Parameters")
ax2.set_xlabel("t")

fig.savefig(save_dir / "spiral_series.svg")
fig2.savefig(save_dir / "posterior_collapse.svg")

fig3, ax3 = plt.subplots(figsize=(8, 2.5))
ax3.hist(
    data[sigmas[-1]]["mu"].reshape(-1),
    bins=np.linspace(-0.1, 0.1, 5),
    label=r"$\mu_\phi(x,y)$",
)
ax3.hist(
    data[sigmas[-1]]["sigma_phi"].reshape(-1),
    bins=np.linspace(0.9, 1.1, 5),
    label=r"$\sigma_\phi(x,y)^2$",
)
ax3.legend()
ax3.set_title("Recognition Model Parameters with Posterior Collapse")

fig.savefig(save_dir / "spiral_series.svg")
# fig2.savefig(save_dir / "posterior_collapse.svg")
fig3.savefig(save_dir / "posterior_collapse_hist.svg")

plt.show()
