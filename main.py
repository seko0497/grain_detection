import torch
import wandb
from config import get_config
from grain_detection.dataset_grain import GrainDataset
from torch.utils.data import DataLoader
from setr import SETR
from train import train


def main():

    config = get_config()

    wandb.init(config=config, entity="seko97", project="grain_detection")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed", 1234))
    torch.backends.cudnn.determerministic = True

    train_loader = DataLoader(GrainDataset(
        config["train_dataset"],
        config["num_data"]
    ), batch_size=config.get("batch_size", 4),
       num_workers=config.get("num_workers", 1),
       persistent_workers=True,
       pin_memory=True)

    model = SETR(
        config["num_patches"],
        config["image_size"],
        config["num_channels"],
        config["embedding_size"],
        config["n_encoder_heads"],
        config["n_encoder_layers"],
        config["decoder_features"],
        config["out_channels"],
        config["decoder_method"]
    )
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)

    model.to(device)

    optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
        model.parameters(), lr=config["learning_rate"]
    )
    loss = getattr(torch.nn, config.get("loss", "CrossEntropyLoss"))()

    wandb.watch(model, log="all")

    for epoch in range(1, config["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch, loss)


if __name__ == '__main__':
    main()
