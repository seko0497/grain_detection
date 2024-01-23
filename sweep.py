from main import main
import wandb

sweep_config = {
    "method": "grid"}

parameters_dict = {
    "batch_size": {
        "values": [16, 32]
        },
    "encoder_type": {
        "values": ["vit_b", "vit_l"]
    },
    "learning_rate": {
        "values": [0.00001, 0.0001]
    },
    "model": {
        "values": ["SETR_PUP", "SETR_MLA"]
    }
}

sweep_config["parameters"] = parameters_dict
sweep_id = wandb.sweep(sweep_config, project="grain_detection")
wandb.agent(sweep_id, main, project="grain_detection")
