import torchaudio
import torch
import numpy as np
import pandas as pd
# from datasets import Dataset
from transformers import WhisperFeatureExtractor
import torch
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    # DataCollatorSpeechSeq2SeqWithPadding,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

raw_path = "/shared/A_track/"
train_json_path = "/shared/A_track/train.json"
dev_json_path = "/shared/A_track/dev_test.json"



train_df = pd.read_json(train_json_path).T

dev_df = pd.read_json(dev_json_path).T



train_df["file_path"] = "processed/"+ train_df["audio_path"] +".mel.pt"
dev_df["file_path"] = "processed/"+ dev_df["audio_path"] +".mel.pt"



dataset = Dataset.from_pandas(train_df)

eval_dataset = Dataset.from_pandas(dev_df)