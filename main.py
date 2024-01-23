""" 
Train a Transformer-based model (SETR or TransUNet) for semantic segmentation
of grain tool images.
"""

import os
import torch
from tqdm import tqdm
import wandb
from config import config as config_dict
from dataset_grain import GrainDataset
from torch.utils.data import DataLoader, ConcatDataset
from models.transunet import TransUNet
from models.setr import SETR
from train import train
from validate import validate
from sklearn.model_selection import train_test_split


def main():

    # initialize weights and biases logging
    if config_dict["use_wandb"]:
        wandb.init(
            config=config_dict, entity="seko97", project="grain_detection")
    config = Config(config=config_dict, wandb=config_dict["use_wandb"])
    
    # check if gpu training is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # make results reproducible
    torch.manual_seed(config.get("random_seed"))
    torch.backends.cudnn.determerministic = True

    # initialize datasets and dataloaders
    train_images, test_images = train_test_split(
        [i for i in range(1, 11) if i != 2],
        test_size=0.1,
        random_state=config.get("random_seed", 1234))

    train_images, validation_images = train_test_split(
        train_images,
        test_size=0.2,
        random_state=config.get("random_seed"))

    train_dataset = GrainDataset(
        config.get("train_dataset"),
        config.get("num_data"),
        in_channels=config.get("in_channels"),
        image_idxs=train_images,)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get("batch_size"),
        num_workers=config.get("num_workers"),
        persistent_workers=True,
        pin_memory=True)
    
    validation_dataset = GrainDataset(
        config.get("train_dataset"),
        config.get("num_data"),
        in_channels=config.get("in_channels"),
        image_idxs=validation_images,
        train=False)
    
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        num_workers=config.get("num_workers"),
        persistent_workers=True,
        pin_memory=True)

    # combine original and with ddpm generated data
    if config.get("frac_original") < 1.0:
        num_original = len(
            os.listdir("/home/kons/02_processed/train/features/1"))
        num_synthetic = num_original * (
            (1.0 - config.get("frac_original")) / config.get("frac_original"))
        synthetic_dataset = GrainDataset(
            config.get("path_synthetic"),
            in_channels=config.get("in_channels"),
            image_idxs=list(range(int(num_synthetic))),
            keys=["T", "F"],
            one_hot=True)

        combined_dataset = ConcatDataset(
            (train_dataset, synthetic_dataset))
        combined_dataloader = DataLoader(
            combined_dataset,
            batch_size=config.get("batch_size"),
            num_workers=config.get("num_workers"),
            persistent_workers=True,
            pin_memory=True,
            shuffle=True)

        train_loader = combined_dataloader

    # load training checkpoint
    if config.get("checkpoint"):
        wandb_api = wandb.Api()
        run = wandb_api.run(config.get('checkpoint'))
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

    # intialize model
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

    # check if training in multiple gpus is possible
    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DataParallel(model)
    # load model checkpoint
    if checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # initialize optimizer
    optimizer = getattr(torch.optim, config.get("optimizer"))(
        model.parameters(), lr=config.get("learning_rate"))
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # initialize loss function
    loss = getattr(torch.nn, config.get("loss"))()

    # log model with wandb
    if config.get("use_wandb"):
        wandb.watch(model, log="all")

    # initialize training
    best = {"epoch": 0, "iou": 0}
    start_epoch = 1
    if checkpoint:
        best["epoch"] = checkpoint["epoch"]
        start_epoch = checkpoint["epoch"] + 1
    del checkpoint

    # start training
    for epoch in tqdm(range(start_epoch, config.get("epochs") + 1)):
        train(
            model, device, train_loader, optimizer, epoch, loss,
            use_wandb=config.get("use_wandb"))
        if epoch % config.get("evaluate_every") == 0:
            iou = validate(
                    model, device, validation_loader, epoch,
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
    """Helper class do switch between casual config and wandb config"""

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
