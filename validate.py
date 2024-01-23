import einops
from matplotlib import cm
import numpy as np
import torch
from tqdm import tqdm
import wandb
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional import dice


def validate(model, device, validation_loader, epoch, use_wandb=False):

    model.eval()

    scores = {"iou": [], "dice": []}
    log_dict = {}

    with torch.no_grad():

        prediction = []
        ground_truth = []

        for batch_idx, batch in enumerate(validation_loader):

            p1 = batch["O"].shape[2] // 256
            p2 = batch["O"].shape[3] // 256

            prediction = []

            for patch in tqdm(batch["I"][0]):

                inp = patch[None].to(device)
                output = model(inp)
                prediction.append(output[0, 0])
            ground_truth = batch["O"][0, 0]

            prediction = einops.rearrange(
                torch.stack(prediction),
                "(p1 p2) h w -> (p1 h) (p2 w)",
                p1=p1,
                p2=p2)

            prediction = torch.round(torch.sigmoid(prediction))

            scores["iou"].append(binary_jaccard_index(
                prediction.cpu(),
                ground_truth.cpu()
            ).item())
            scores["dice"].append(dice(
                prediction.cpu(),
                ground_truth.cpu().int()
            ).item())

            log_dict[f"val_image_pred{batch_idx}"] = wandb.Image(
                get_rgb(prediction))
            log_dict[f"val_image_target{batch_idx}"] = wandb.Image(
                get_rgb(ground_truth.float()))

        log_dict["val_iou_mean"] = np.mean(scores["iou"])
        log_dict["val_dice_mean"] = np.mean(scores["dice"])
        for i in range(len(validation_loader)):
            log_dict[f"val_iou_{i}"] = scores["iou"][i]
            log_dict[f"val_dice_{i}"] = scores["dice"][i]

        if use_wandb:
            wandb.log(log_dict, step=epoch)

        return scores["iou"][0]


def get_rgb(image, colomap="viridis"):

    rgb_image = image.cpu().detach().numpy()
    cmap = cm.get_cmap(colomap)
    rgb_image = cmap(rgb_image)[..., :3]

    return rgb_image
