from vae_demo.config import VAEConfig
from vae_demo.model import train, reparam
import matplotlib.pyplot as plt
import torch
from vae_demo.spiral import Spiral
import numpy as np


spread = 0.1
sigmas = [0.0001, 0.1, 0.4]
data = {}
for sigma in sigmas:
    cur_data = {}
    cfg = VAEConfig(
        spread=spread,
        decoder_sigma=sigma,
        omega=3 * np.pi,
        epochs=15000,
        batch_size=100,
        t_steps=150,
        t_start=0.2,
        encoder_hidden_layers=2,
        decoder_hidden_layers=2,
    )
    encoder, decoder, kl_loss_hist, re_loss_hist, loss_hist, dset = train(cfg)
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
    }
    data[sigma] = cur_data


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(12, 5), sharey=True)
for i, sigma in enumerate(data.keys()):
    _data = data[sigma]
    dset = _data["dset"]
    ts = _data["dset"].ts
    spiral = Spiral(omega=dset.omega, rdot=dset.rdot, spread=dset.spread)
    exact_x, exact_y = spiral.exact(ts)
    ax[i].plot(exact_x, exact_y)
    ax[i].plot(dset[:, 0], dset[:, 1], ".", color="black", alpha=0.5)
    ax[i].plot(
        _data["output"][:, 0], _data["output"][:, 1], "x", color="red", alpha=0.5
    )
    ax[i].set_xticks([])
    ax[i].set_yticks([])
    ax[i].set_title(r"$\sigma / \sigma_{data} = $" + f"{sigma / spread}")
fig.tight_layout()
fig2, ax2 = plt.subplots(ncols=3, nrows=1, figsize=(12, 5))
for i, sigma in enumerate(data.keys()):
    _data = data[sigma]
    ax2[i].plot(np.array(_data["re"]) * sigma**2)
    ax2[i].plot(_data["kl"])
    ax2[i].set_yscale("log")
#
#    ax[2, i].scatter(
#        _data["mu"].reshape(-1),
#        np.zeros_like(_data["mu"]).reshape(-1),
#        c=_data["dset"].ts.numpy().reshape(-1),
#        cmap="viridis",
#    )
plt.show()
