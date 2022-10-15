import einops
import numpy as np
import torch
from tqdm import tqdm
import wandb
from torchmetrics.functional.classification import binary_jaccard_index
from torchmetrics.functional import dice


def validate(model, device, validation_loader, epoch):

    model.eval()

    scores = {"iou": [], "dice": []}
    log_dict = {}

    with torch.no_grad():

        for batch_idx, batch in enumerate(validation_loader):

            p1 = batch["O"].shape[2] // 256
            p2 = batch["O"].shape[3] // 256

            features = einops.rearrange(
                batch["I"][0],
                "(p1 p2) c h w -> c (p1 h) (p2 w)",
                p1=p1,
                p2=p2
            )

            prediction = []

            for patch in tqdm(batch["I"][0]):

                inp = patch[None].to(device)
                output = model(inp)
                prediction.append(output[0, 0])

            prediction = einops.rearrange(
                torch.stack(prediction),
                "(p1 p2) h w -> (p1 h) (p2 w)",
                p1=p1,
                p2=p2
            )
            prediction = torch.round(torch.sigmoid(prediction))

            scores["iou"].append(binary_jaccard_index(
                prediction.cpu(),
                batch["O"][0, 0].cpu()
            ).item())
            scores["dice"].append(dice(
                prediction.cpu(),
                batch["O"][0, 0].cpu().int()
            ).item())

            log_dict[f"val_image_{batch_idx}"] = wandb.Image(
                features[0].cpu().detach().numpy(),
                masks={
                    "predictions": {
                        "mask_data": torch.round(torch.sigmoid(
                            prediction)).cpu().detach().numpy()},
                    "ground_truth": {
                        "mask_data": batch["O"][0, 0].cpu().detach().numpy()}
                }
            )

        log_dict["val_iou_mean"] = np.mean(scores["iou"])
        log_dict["val_dice_mean"] = np.mean(scores["dice"])
        for i in range(len(validation_loader)):
            log_dict[f"val_iou_{i}"] = scores["iou"][i]
            log_dict[f"val_dice_{i}"] = scores["dice"][i]

        wandb.log(log_dict, step=epoch)
