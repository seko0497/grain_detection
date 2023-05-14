from torch import embedding
import wandb


# Data config

train_dataset = "data/grains"
felix_data = True

checkpoint = None

# Model config

vit_defaults = {
    "vit_b": {
        "embedding_size": 768,
        "n_encoder_heads": 12,
        "n_encoder_layers": 12,
        "dim_mlp": 3072},
    "vit_l": {
        "embedding_size": 1024,
        "n_encoder_heads": 16,
        "n_encoder_layers": 24,
        "dim_mlp": 4096}}
setr_defaults = {
    "pup_features": [1024, 256, 256, 256],
    "n_mla_heads": 4
}

num_patches = (16, 16)
image_size = (256, 256)
in_channels = ["intensity", "depth"]
model = "TransUNet"
encoder_type = "vit_b"
out_channels = 1

# Train config

num_data = 5000
batch_size = 32
optimizer = "Adam"
loss = "BCEWithLogitsLoss"
learning_rate = 0.0001
epochs = 200
num_workers = 32

# Combining

frac_original = 0.1
path_synthetic = "grain_generation/samples/youthful-paper-47/epoch510_steps100"

# Eval config

evaluate_every = 1

random_seed = 1234
use_wandb = True

config = {
    "train_dataset": train_dataset,
    "num_data": num_data,
    "num_patches": num_patches,
    "image_size": image_size,
    "in_channels": in_channels,
    "model": model,
    "vit_defaults": vit_defaults,
    "setr_defaults": setr_defaults,
    "encoder_type": encoder_type,
    "out_channels": out_channels,
    "batch_size": batch_size,
    "optimizer": optimizer,
    "loss": loss,
    "random_seed": random_seed,
    "epochs": epochs,
    "num_workers": num_workers,
    "frac_original": frac_original,
    "path_synthetic": path_synthetic,
    "learning_rate": learning_rate,
    "evaluate_every": evaluate_every,
    "use_wandb": use_wandb,
    "felix_data": felix_data,
    "checkpoint": checkpoint
}
