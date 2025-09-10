import json
import random

def split_train_val(input_path, train_out_path, val_out_path, val_ratio=0.1, seed=42):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.seed(seed)
    random.shuffle(lines)

    val_size = int(len(lines) * val_ratio)
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]

    with open(train_out_path, 'w', encoding='utf-8') as f_train, \
         open(val_out_path, 'w', encoding='utf-8') as f_val:
        f_train.writelines(train_lines)
        f_val.writelines(val_lines)

    print(f"Split completed: {len(train_lines)} train, {len(val_lines)} val")

# Example usage
# split_train_val(
#     input_path="data/train_nemo.json",
#     train_out_path="data/train_split.json",
#     val_out_path="data/val_split.json",
#     val_ratio=0.1
# )

split_train_val(
    input_path="/A_track/train_nemo.json",
    train_out_path="/A_track/train_split_nemo.json",
    val_out_path="/A_track/val_split_nemo.json",
    val_ratio=0.1
)