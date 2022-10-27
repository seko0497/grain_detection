from sqlalchemy import values
from main import main
import wandb

sweep_config = {
    "method": "random",
}

parameters_dict = {
    "batch_size": {
        "values": [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        },
    "encoder_type": {
        "values": ["vit_b", "vit_l"]
    },
    "learning_rate": {
        "values": [0.00001, 0.0001, 0.001, 0.01, 0.1]
    }
}

sweep_config["parameters"] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="grain_detection")
wandb.agent(sweep_id, main, count=30, project="grain_detection")
