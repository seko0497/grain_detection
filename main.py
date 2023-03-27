from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import wandb
from config import get_config
from grain_detection.dataset_grain import GrainDataset
from torch.utils.data import DataLoader
from grain_detection.transunet import TransUNet
from setr import SETR
from train import train
from validate import validate
from sklearn.model_selection import train_test_split
import importlib


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
        random_state=1  # =config.get("random_seed", 1234)
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

    if not config["felix_data"]:
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

    else:

        graindetection_felix = importlib.import_module(
            "lib.grain-detection.graindetection.dataprocessor")
        GrainDatasetFelix = graindetection_felix.GrainDataset
        if use_wandb:
            train_loader = DataLoader(GrainDatasetFelix(
                "/home/kons/02_processed/train/features",
                "/home/kons/02_processed/train/target"
            ), batch_size=wandb.config.batch_size,
             num_workers=config.get("num_workers", 1),
             persistent_workers=True,
             pin_memory=True)
        else:
            train_loader = DataLoader(GrainDatasetFelix(
                "/home/kons/02_processed/train/features",
                "/home/kons/02_processed/train/target"
            ), batch_size=config.get("batch_size", 4),
             num_workers=config.get("num_workers", 1),
             persistent_workers=True,
             pin_memory=True)
        validation_loader = DataLoader(GrainDatasetFelix(
            "/home/kons/02_processed/test/features",
            "/home/kons/02_processed/test/target"
        ), batch_size=1,
         num_workers=config.get("num_workers", 1),
         persistent_workers=True,
         pin_memory=True)

    # load Checkpoint
    if config.get("checkpoint"):
        wandb_api = wandb.Api()
        run = wandb_api.run(config['checkpoint'])
        run_name = run.name
        model_folder = f"grain_detection/models/{run.name}"
        print(f"restoring {model_folder}")
        checkpoint = wandb.restore(
            "grain_detection/best.pth",
            run_path=config["checkpoint"],
            root=model_folder)
        print("restored")
        checkpoint = torch.load(checkpoint.name)
    else:
        checkpoint = None

    if use_wandb and wandb.config.encoder_type == "vit_b":

        wandb.config.update(
            {
                "embedding_size": 768,
                "n_encoder_heads": 12,
                "n_encoder_layers": 12,
                "dim_mlp": 3072
            }, allow_val_change=True)

    elif use_wandb and config["encoder_type"] == "vit_l":

        wandb.config.update(
            {
                "embedding_size": 1024,
                "n_encoder_heads": 16,
                "n_encoder_layers": 24,
                "dim_mlp": 4096
            }, allow_val_change=True)

    if use_wandb:

        model = TransUNet(
            wandb.config.embedding_size,
            wandb.config.n_encoder_heads,
            wandb.config.n_encoder_layers,
            wandb.config.out_channels,
            wandb.config.dim_mlp
        )

    else:

        model = TransUNet(
            config["embedding_size"],
            config["n_encoder_heads"],
            config["n_encoder_layers"],
            config["out_channels"],
            config["dim_mlp"]
        )

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    if use_wandb:

        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(), lr=wandb.config.learning_rate
        )

    else:
        optimizer = getattr(torch.optim, config.get("optimizer", "Adam"))(
            model.parameters(), lr=config["learning_rate"]
        )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss = getattr(torch.nn, config.get("loss", "CrossEntropyLoss"))()

    if use_wandb:
        wandb.watch(model, log="all")

    best = {"epoch": 0, "iou": 0}
    start_epoch = 1
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"] + 1
    del checkpoint
    for epoch in tqdm(range(start_epoch, config["epochs"] + 1)):
        train(
            model, device, train_loader, optimizer, epoch, loss,
            felix_data=config["felix_data"], use_wandb=use_wandb)
        if epoch % config["evaluate_every"] == 0:
            iou = validate(
                    model, device, validation_loader, epoch,
                    felix_data=config["felix_data"], use_wandb=use_wandb)
            if iou > best["iou"]:
                best["iou"] = iou
                best["epoch"] = epoch
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'iou': iou,
                        'loss': loss}, f"grain_detection/best.pth")
                if config.get("use_wandb"):
                    wandb.save("grain_detection/best.pth")
            wandb.log({"best_epoch": best["epoch"], "best_iou": best["iou"]})


if __name__ == '__main__':
    main()
