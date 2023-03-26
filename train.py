from torch.utils.data import Dataset, DataLoader
from grain_detection.dataset_grain import GrainDataset
from setr import SETR
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import torch


def train(
        model, device, data_loader, optimizer,
        epoch, loss_fn, felix_data=False, use_wandb=False):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        if not felix_data:
            inp, trg = batch["I"].to(device), batch["O"].to(device)
        else:
            inp, trg = batch["F"].to(device), batch["T"].to(device)
            trg = torch.argmax(trg, dim=1, keepdim=True).float()

        # Forward pass
        optimizer.zero_grad()
        output = model(inp)

        # Backward pass
        loss = loss_fn(output, trg)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx == 0:

            if use_wandb:

                wandb.log({"Train samples": wandb.Image(
                    inp[0, 0].cpu().detach().numpy(),
                    masks={
                        "predictions": {
                            "mask_data": torch.round(torch.sigmoid(
                                output[0, 0])).cpu().detach().numpy()},
                        "ground_truth": {
                            "mask_data": trg[0, 0].cpu().detach().numpy()}
                    }
                )}, step=epoch, commit=False)

    epoch_loss /= len(data_loader)
    wandb.log({"train_loss": epoch_loss}, commit=False)
