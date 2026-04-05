import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from vae_demo.spiral import SpiralDataset


torch.manual_seed(42)


class SpiralEncoder(nn.Module):
    """
    Encode 2D spiral data points encoded according to recognition model
    q_phi(z|x). q_phi(z|x) is assumed normal, and the encoder predicts
    the mean and log variance per data point x.
    """

    def __init__(self, latent_dims=2, hidden_dim=32, hidden_layers=1):
        super().__init__()
        self.latent_dims = latent_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        self.h = nn.Sequential(*layers)
        self.linear_mu = nn.Linear(self.hidden_dim, self.latent_dims)
        # mlp_sigma predicts the log variance, not the variance or std
        self.linear_sigma = nn.Linear(self.hidden_dim, self.latent_dims)

    def forward(self, input):
        h = self.h(input)
        mu = self.linear_mu(h)
        sigma = self.linear_sigma(h)
        return mu, sigma


def reparam(mu, logsigma):
    """
    Reparamaterization trick to backprop through sampling q_phi(z|x).
    mu, logsigma are the outputs of the encoder. logsigma is treated
    as being log (sigma^2).
    """
    samples = torch.randn_like(logsigma)
    sigma = torch.exp(0.5 * logsigma)
    return mu + sigma * samples


class SpiralDecoder(nn.Module):
    """
    Decode 2D spiral data points from latent variable z according to a
    generative model p_theta(x|z). p_theta(x|z) is assumed to be normal
    with mu per latent instance z and variance fixed. In other words, the
    model simply outputs mu_theta(z) due to our assumptions.
    """

    def __init__(self, latent_dims=2, hidden_dim=32, hidden_layers=1):
        super().__init__()
        self.latent_dims = latent_dims
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(self.latent_dims, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 2)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, z):
        return self.mlp(z)


def reconstruction_loss(x, mu, sigma=1.0):
    """
    Compute the reconstruction loss log p_theta(x|z). We are assuming
    p_theta(x|z) ~ N(x; mu(theta), sigma^2) where mu(theta) is the output
    of the decoder, sigma is fixed, so the second term in the loss of equation
    10 in Kingma paper is a MSE loss SUM(-1/L (1/2*sigma^2) (x - mu)**2)
    """
    per_dim = (x - mu) ** 2
    per_datapoint = per_dim.sum(dim=1)
    return per_datapoint.mean() / (2 * sigma**2)


def KL_loss(mu, logsigma):
    """
    mu: encoder predicted mu_phi(z|x)
    logsigma: encoder predicted log(sigma_phi(z|x)^2)
    """
    sigma_squared = logsigma.exp()
    per_dim = 0.5 * (1 + logsigma - mu**2 - sigma_squared)
    per_datapoint = per_dim.sum(dim=1)
    return -per_datapoint.mean()


def train():
    dset = SpiralDataset(np.linspace(0.1, 1, 100), spread=0.01, omega=4 * np.pi)
    data_loader = DataLoader(dset, shuffle=True, batch_size=100)
    encoder = SpiralEncoder(latent_dims=1, hidden_dim=64, hidden_layers=2)
    decoder = SpiralDecoder(latent_dims=1, hidden_dim=64, hidden_layers=2)
    lr_enc = 1e-3
    lr_dec = 1e-3
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=lr_enc)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=lr_dec)

    epochs = 50000

    kl_loss_hist = []
    re_loss_hist = []
    loss_hist = []

    for epoch in range(epochs):
        kl_weight = 1.0
        for batch in data_loader:
            optimizer_enc.zero_grad()
            optimizer_dec.zero_grad()

            mu, logsigma = encoder(batch)
            z = reparam(mu, logsigma)
            decoded = decoder(z)
            kl_loss = KL_loss(mu, logsigma)
            re_loss = reconstruction_loss(batch, decoded, sigma=0.01)
            loss = kl_weight * kl_loss + re_loss
            loss.backward()
            optimizer_enc.step()
            optimizer_dec.step()
            kl_loss_hist.append(kl_loss.item())
            re_loss_hist.append(re_loss.item())
            loss_hist.append(loss.item())
        if epoch % 100 == 0:
            print(
                f"Epoch: {epoch} | "
                f"Loss (Total): {loss.item()} | "
                f"Loss (KL): {kl_loss.item()} | "
                f"Loss (Re): {re_loss.item()} | "
            )
    return encoder, decoder, kl_loss_hist, re_loss_hist, loss_hist


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    encoder, decoder, kl_loss_hist, re_loss_hist, loss_hist = train()
    dset = SpiralDataset(np.linspace(0.1, 1, 100), spread=0.01, omega=4 * np.pi)
    input = dset.data
    with torch.no_grad():
        mu, logsigma = encoder(input)
        z = reparam(mu, logsigma)
        output = decoder(z)

    plt.figure()
    plt.plot(kl_loss_hist)
    plt.plot(re_loss_hist)
    plt.yscale("log")

    plt.figure()
    plt.plot(loss_hist)
    plt.yscale("log")

    plt.figure()
    plt.plot(dset[:, 0], dset[:, 1], ".")
    plt.plot(output[:, 0], output[:, 1], "x", color="red", alpha=0.7)

    plt.figure()
    plt.plot(dset[:, 0], dset[:, 1])
    plt.plot(output[:, 0], output[:, 1], "--", color="red", alpha=0.7)

    with torch.no_grad():
        mu, logsigma = encoder(dset.data)

    plt.figure()
    plt.scatter(
        mu.numpy(),
        np.zeros_like(mu.numpy()),
        c=np.linspace(0.1, 1, 100),
        cmap="viridis",
    )
    plt.colorbar(label="t")
    plt.title("Latent space")

    plt.show()
