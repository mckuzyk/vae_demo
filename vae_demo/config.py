from dataclasses import dataclass, asdict
import json
from numpy import pi


@dataclass
class VAEConfig:
    # Global parameters
    latent_dims: int = 1

    # Encoder parameters
    encoder_hidden_dim: int = 32
    encoder_hidden_layers: int = 1

    # Decoder parameters
    decoder_hidden_dim: int = 32
    decoder_hidden_layers: int = 1

    decoder_sigma: float = 0.01

    # Training parameters
    epochs: int = 20000
    lr_enc: float = 1e-3
    lr_dec: float = 1e-3
    batch_size: int = 100
    shuffle_batch: bool = True

    # Training data parameters
    omega: float = 2 * pi
    rdot: float = 1.0
    spread: float = 0.01
    t_start: float = 0.1
    t_stop: float = 1.0
    t_steps: int = 100

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
