import json
import os

# --- Configuration Section ---

# 1. Project and Data Paths
# This should be the root folder of your project.
PROJECT_ROOT = "." 
# The name of your JSON file with audio paths and transcripts.
TRAIN_JSON_PATH = os.path.join(PROJECT_ROOT, "train.json") 
# The folder where all your .wav files are stored.
WAV_DIR = os.path.join(PROJECT_ROOT, "wavs") 




EXP_NAME = "whisper_finetune_experiment"


# --- End of Configuration Section ---

import os
from glob import glob

import numpy as np
import librosa

import torch
from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch   
from espnet2.layers.create_adapter_fn import create_lora_adapter
import espnetez as ez

# Define hyper parameters
DUMP_DIR = f"./dump"
CSV_DIR = f"./transcription"
EXP_DIR = f"./exp/finetune"
STATS_DIR = f"./exp/stats_finetune"

FINETUNE_MODEL =  "espnet/owsm_ctc_v4_1B" # "espnet/owsm_v3.1_ebf"
LORA_TARGET = [
    "w_1", "w_2", "merge_proj"
]


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
pretrained_model = Speech2TextGreedySearch.from_pretrained(
    FINETUNE_MODEL,
        device='cuda' if torch.cuda.is_available() else 'cpu',
   
) # Load model to extract configs.
pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter
del pretrained_model


# For the configuration, please refer to the last cell in this notebook.
finetune_config = ez.config.update_finetune_config(
	's2t',
	pretrain_config,
	f"finetune_with_lora.yaml"
)


import pickle

# Save finetune_config to a pickle file
with open('finetune_config.pkl', 'wb') as f:
    pickle.dump(finetune_config, f)

print("Finetune config saved to finetune_config.pkl")