from torch.utils.data import Dataset, DataLoader
from grain_detection.dataset_grain import GrainDataset
from setr import SETR
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import torch


def train(model, device, data_loader, optimizer, epoch, loss_fn):

    model.train()

    epoch_loss = 0

    for batch_idx, batch in enumerate(tqdm(data_loader)):

        inp, trg = batch["I"].to(device), batch["O"].to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(inp)

        # Backward pass
        loss = loss_fn(output, trg)

        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx == 0:

            wandb.log({"intensity": wandb.Image(
                inp[0, 0].cpu().detach().numpy(),
                masks={
                    "predictions": {
                        "mask_data": torch.round(
                            output[0, 0]).cpu().detach().numpy()},
                    "ground_truth": {
                        "mask_data": trg[0, 0].cpu().detach().numpy()}
                }
            )}, step=epoch)

    epoch_loss /= len(data_loader)
    wandb.log({"train_loss": epoch_loss})
