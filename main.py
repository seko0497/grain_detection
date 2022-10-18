import torch
import wandb
from config import get_config
from grain_detection.dataset_grain import GrainDataset
from torch.utils.data import DataLoader
from setr import SETR
from train import train
from validate import validate
from sklearn.model_selection import train_test_split


def main():

    config = get_config()

    use_wandb = config.get("use_wandb", False)

    if use_wandb:
        wandb.init(config=config, entity="seko97", project="grain_detection")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed", 1234))
    torch.backends.cudnn.determerministic = True

    train_images, test_images = train_test_split(
        [i for i in range(1, 11) if i != 2],
        test_size=0.1,
        random_state=config.get("random_seed", 1234)
    )

    train_images, validation_images = train_test_split(
        train_images,
        test_size=0.2,
        random_state=config.get("random_seed", 1234)
    )

    if use_wandb:
        wandb.config.train_images = train_images
        wandb.config.validation_images = validation_images
        wandb.config.test_images = test_images

    if use_wandb:
        train_loader = DataLoader(GrainDataset(
            config["train_dataset"],
            config["num_data"],
            in_channels=config["in_channels"],
            image_idxs=train_images,
        ), batch_size=wandb.config.batch_size,
           num_workers=config.get("num_workers", 1),
           persistent_workers=True,
           pin_memory=True)

    else:

        train_loader = DataLoader(GrainDataset(
            config["train_dataset"],
            config["num_data"],
            in_channels=config["in_channels"],
            image_idxs=train_images,
        ), batch_size=config.get("batch_size", 4),
           num_workers=config.get("num_workers", 1),
           persistent_workers=True,
           pin_memory=True)

    validation_loader = DataLoader(GrainDataset(
        config["train_dataset"],
        config["num_data"],
        in_channels=config["in_channels"],
        image_idxs=validation_images,
        train=False
    ), batch_size=1,
       num_workers=config.get("num_workers", 1),
       persistent_workers=True,
       pin_memory=True)

    if use_wandb:

        model = SETR(
            wandb.config.num_patches,
            wandb.config.image_size,
            len(wandb.config.in_channels),
            wandb.config.embedding_size,
            wandb.config.n_encoder_heads,
            wandb.config.n_encoder_layers,
            [
                wandb.config.embedding_size,
                wandb.config.embedding_size // 2,
                wandb.config.embedding_size // 2,
                wandb.config.embedding_size // 2
            ],
            wandb.config.out_channels,
            wandb.config.decoder_method
        )

    else:

        model = SETR(
            config["num_patches"],
            config["image_size"],
            len(config["in_channels"]),
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

    if use_wandb:

        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(), lr=wandb.config.learning_rate
        )

    else:
        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(), lr=config["learning_rate"]
        )

    loss = getattr(torch.nn, config.get("loss", "CrossEntropyLoss"))()

    if use_wandb:
        wandb.watch(model, log="all")

    for epoch in range(1, config["epochs"] + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        if epoch % config["evaluate_every"] == 0:
            validate(model, device, validation_loader, epoch)


if __name__ == '__main__':
    main()
