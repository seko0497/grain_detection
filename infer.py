import importlib
import os
import numpy as np
import torch
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
from PIL import Image

from transunet import TransUNet

run_path = "seko97/grain_detection/n1tnovoc"
felix_data = True
num_workers = 32
save_patches = True
data_type = "NPY"

wandb_api = wandb.Api()
run = wandb_api.run(run_path)
run_name = run.name

model_folder = f"grain_detection/models/{run_name}"
print(f"restoring {model_folder}")
checkpoint = wandb.restore(
    "grain_detection/best.pth",
    run_path=run_path,
    root=model_folder)
print("restored")
checkpoint = torch.load(checkpoint.name)

model = TransUNet(
            run.config["embedding_size"],
            run.config["n_encoder_heads"],
            run.config["n_encoder_layers"],
            run.config["out_channels"],
            run.config["dim_mlp"]
        )

model = torch.nn.parallel.DataParallel(model)
model.load_state_dict(checkpoint["model_state_dict"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

save_folder = f"grain_detection/samples/{run_name}/{checkpoint['epoch']}"

if felix_data:
    graindetection_felix = importlib.import_module(
            "lib.grain-detection.graindetection.dataprocessor")
    GrainDatasetFelix = graindetection_felix.GrainDataset
    validation_loader = DataLoader(GrainDatasetFelix(
            "/home/kons/02_processed/test/features",
            "/home/kons/02_processed/test/target"
        ), batch_size=1,
         num_workers=num_workers,
         persistent_workers=True,
         pin_memory=True)

    for i, batch in enumerate(tqdm(validation_loader)):

        if save_patches:

            p1 = 4
            p2 = 42
            output = model(batch["F"].to(device))
            prediction = (output[0, 0])
            prediction = torch.round(torch.sigmoid(prediction))
            prediction = prediction.cpu().detach().numpy()
            if data_type == "PNG":
                prediction = (prediction * 255).astype(np.uint8)
                image = Image.fromarray(prediction)
                if not os.path.exists(f"{save_folder}/png"):
                    os.makedirs(f"{save_folder}/png")
                image.save(
                    f"{save_folder}/png/pred_{str(i).zfill(3)}.png")
            elif data_type == "NPY":
                if not os.path.exists(f"{save_folder}/npy"):
                    os.makedirs(f"{save_folder}/npy")
                np.save(
                    f"{save_folder}/npy/pred_{str(i).zfill(3)}.npy",
                    prediction)
