from dataclasses import dataclass, asdict
import json


@dataclass
class VAEConfig:
    latent_dims: int = 1
    # Encoder parameters
    encoder_hidden_dim: int = 32
    encoder_hidden_layers: int = 1

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
