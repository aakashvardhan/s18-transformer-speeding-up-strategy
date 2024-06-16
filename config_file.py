from pathlib import Path
import torch

def get_config():
    return {
        "batch_size": 128,
        "precision": "16-mixed",
        "accelerator": "cuda",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "progress_bar_refresh_rate": 10,
        "num_epochs": 18,
        "num_iter": 100,
        "lr": 0.0003,
        "seq_len": 160,
        "n_workers": 4,
        "num_examples": 5,
        "one_cycle_best_lr":0.00001,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "ckpt_path": "default",
        "model_basename": "tmodel_",
        "preload": False,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"}

def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)