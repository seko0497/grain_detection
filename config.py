from torch import embedding
import wandb


def get_config():

    # Data config

    train_dataset = "data/grains"

    # Model config

    num_patches = (16, 16)
    image_size = (256, 256)
    num_channels = 2
    embedding_size = 256
    n_encoder_heads = 1
    n_encoder_layers = 20
    decoder_features = [512, 128, 128, 128]
    decoder_method = "PUP"
    out_channels = 1

    # Train config

    num_data = 5000
    batch_size = 16
    optimizer = "Adam"
    loss = "BCELoss"
    learning_rate = 0.01
    epochs = 100
    num_workers = 16

    random_seed = 1234

    config = {
        "train_dataset": train_dataset,
        "num_data": num_data,
        "num_patches": num_patches,
        "image_size": image_size,
        "num_channels": num_channels,
        "embedding_size": embedding_size,
        "n_encoder_heads": n_encoder_heads,
        "n_encoder_layers": n_encoder_layers,
        "decoder_features": decoder_features,
        "decoder_method": decoder_method,
        "out_channels": out_channels,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss": loss,
        "random_seed": random_seed,
        "epochs": epochs,
        "num_workers": num_workers,
        "learning_rate": learning_rate
    }

    return config
