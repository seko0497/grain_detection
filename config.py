from torch import embedding
import wandb


def get_config():

    # Data config

    train_dataset = "data/grains"

    # Model config

    num_patches = (16, 16)
    image_size = (256, 256)
    in_channels = ["intensity", "depth"]
    embedding_size = 768
    n_encoder_heads = 12
    n_encoder_layers = 12
    dim_mlp = 3072
    encoder_type = "custom"
    decoder_features = [1024, 256, 256, 256]
    decoder_method = "MLA"
    n_mla_heads = 4
    out_channels = 1

    # Train config

    num_data = 5000
    batch_size = 16
    optimizer = "Adam"
    loss = "BCEWithLogitsLoss"
    learning_rate = 0.00001
    epochs = 50
    num_workers = 8

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
        "embedding_size": embedding_size,
        "n_encoder_heads": n_encoder_heads,
        "n_encoder_layers": n_encoder_layers,
        "dim_mlp": dim_mlp,
        "encoder_type": encoder_type,
        "decoder_features": decoder_features,
        "decoder_method": decoder_method,
        "n_mla_heads": n_mla_heads,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss,
        "random_seed": random_seed,
        "epochs": epochs,
        "num_workers": num_workers,
        "learning_rate": learning_rate,
        "evaluate_every": evaluate_every,
        "use_wandb": use_wandb
    }

    return config
