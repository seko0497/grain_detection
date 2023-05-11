from matplotlib import pyplot as plt
import torch
from tqdm import tqdm
import wandb
from config import config as config_dict
from grain_detection.dataset_grain import GrainDataset
from torch.utils.data import DataLoader
from grain_detection.transunet import TransUNet
from setr import SETR
from train import train
from validate import validate
from sklearn.model_selection import train_test_split
import importlib


def main():

    # initialize weights and biases logging
    if config_dict["use_wandb"]:
        wandb.init(
            config=config_dict, entity="seko97", project="grain_detection")
    config = Config(config=config_dict, wandb=config_dict["use_wandb"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(config.get("random_seed"))
    torch.backends.cudnn.determerministic = True

    if not config.get("felix_data"):

        train_images, test_images = train_test_split(
            [i for i in range(1, 11) if i != 2],
            test_size=0.1,
            random_state=1  # =config.get("random_seed", 1234)
        )

        train_images, validation_images = train_test_split(
            train_images,
            test_size=0.2,
            random_state=config.get("random_seed")
        )

        train_loader = DataLoader(GrainDataset(
            config.get("train_dataset"),
            config.get("num_data"),
            in_channels=config.get("in_channels"),
            image_idxs=train_images,
        ), batch_size=config.get("batch_size"),
            num_workers=config.get("num_workers"),
            persistent_workers=True,
            pin_memory=True)
        validation_loader = DataLoader(GrainDataset(
            config.get("train_dataset"),
            config.get("num_data"),
            in_channels=config.get("in_channels"),
            image_idxs=validation_images,
            train=False
        ), batch_size=1,
         num_workers=config.get("num_workers"),
         persistent_workers=True,
         pin_memory=True)

    else:

        graindetection_felix = importlib.import_module(
            "lib.grain-detection.graindetection.dataprocessor")
        GrainDatasetFelix = graindetection_felix.GrainDataset

        train_loader = DataLoader(GrainDatasetFelix(
            "/home/kons/02_processed/train/features",
            "/home/kons/02_processed/train/target"
        ), batch_size=config.get("batch_size"),
            num_workers=config.get("num_workers"),
            persistent_workers=True,
            pin_memory=True)
        validation_loader = DataLoader(GrainDatasetFelix(
            "/home/kons/02_processed/test/features",
            "/home/kons/02_processed/test/target"
        ), batch_size=1,
         num_workers=config.get("num_workers"),
         persistent_workers=True,
         pin_memory=True)

    # load Checkpoint
    if config.get("checkpoint"):
        wandb_api = wandb.Api()
        run = wandb_api.run(config.get('checkpoint'))
        run_name = run.name
        model_folder = f"grain_detection/models/{run.name}"
        print(f"restoring {model_folder}")
        checkpoint = wandb.restore(
            "grain_detection/best.pth",
            run_path=config.get("checkpoint"),
            root=model_folder)
        print("restored")
        checkpoint = torch.load(checkpoint.name)
    else:
        checkpoint = None

    vit_config = config.get("vit_defaults")[config.get("encoder_type")]

    if config.get("model") == "TransUNet":
        model = TransUNet(
            **vit_config,
            out_channels=config.get("out_channels")
        )
    elif "SETR" in config.get("model"):
        decoder_method = config.get("model").split("_")[1]
        model = SETR(
            config.get("num_patches"),
            config.get("image_size"),
            len(config.get("in_channels")),
            vit_config["embedding_size"],
            vit_config["n_encoder_heads"],
            vit_config["n_encoder_layers"],
            vit_config["dim_mlp"],
            config.get("encoder_type"),
            config.get("setr_defaults")["pup_features"],
            config.get("out_channels"),
            decoder_method,
            config.get("setr_defaults")["n_mla_heads"])

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)

    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(), lr=config.get("learning_rate"))

    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    loss = getattr(torch.nn, config.get("loss"))()

    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    best = {"epoch": 0, "iou": 0}
    start_epoch = 1
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"] + 1
    del checkpoint
    for epoch in tqdm(range(start_epoch, config.get("epochs") + 1)):
        train(
            model, device, train_loader, optimizer, epoch, loss,
            felix_data=config.get("felix_data"),
            use_wandb=config.get("use_wandb"))
        if epoch % config.get("evaluate_every") == 0:
            iou = validate(
                    model, device, validation_loader, epoch,
                    felix_data=config.get("felix_data"),
                    use_wandb=config.get("use_wandb"))
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


class Config():

    def __init__(self, config, wandb=True):

        self.use_wand = wandb
        self.config = config

    def get(self, key):

        if self.use_wand:

            return getattr(wandb.config, key)

        else:
            return self.config[key]


if __name__ == '__main__':
    main()
