import einops
import torch
from tqdm import tqdm
import wandb


def validate(model, device, validation_loader, epoch):

    model.eval()

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

            wandb.log({"Validation": wandb.Image(
                features[0].cpu().detach().numpy(),
                masks={
                    "predictions": {
                        "mask_data": torch.round(torch.sigmoid(
                            prediction)).cpu().detach().numpy()},
                    "ground_truth": {
                        "mask_data": batch["O"][0, 0].cpu().detach().numpy()}
                }
            )}, step=epoch)
