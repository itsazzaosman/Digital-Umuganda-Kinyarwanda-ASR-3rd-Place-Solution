import re
import os

import torch 
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def standardize_quotation(train_df):
    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "\n": '',
        u'\xa0': u' ',

        '，':',',

        '&': "uye",
        '<': "",
        '*': "",
        '>': "",
        '#': "",
        '…': ".",
        '．': ".",
        '+': "",
        '=': "",
        '≠':'',
        '[': "(",
        ']': ")",
        '_':'-',

        'é': 'e',
        'ü': 'u',
        'ì': 'i',
        'ķ': 'k',
        'è': 'e',
    }
    pattern = re.compile("|".join(map(re.escape, replacements.keys())))
    train_df["transcription"] = train_df["transcription"].str.replace(
        pattern, lambda m: replacements[m.group()], regex=True
    )
    return train_df[["transcription"]]



 # integrated_data_preparation.py
import json
import os
import re
import random
from tqdm import tqdm
from nemo.collections.asr.parts.utils.manifest_utils import read_manifest, write_manifest

# --- Part 1: Convert Initial JSON to NeMo Manifest ---
def convert_to_nemo_manifest(config):
    """Converts a project-specific JSON file to a NeMo-compatible JSON manifest."""
    input_path = config['initial_json_path']
    output_path = config['raw_nemo_manifest_path']
    audio_base_path = config['audio_base_path']
    
    print(f"Step 1: Converting {input_path} to NeMo manifest format...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    manifest_lines = []
    for key, entry in tqdm(data.items(), total=len(data), desc="Converting to manifest"):
        try:
            audio_file = entry['audio_path'].replace('audio/', '') + '.wav'
            audio_path = os.path.join(audio_base_path, audio_file)
            transcription = entry['transcription']
            duration = entry['duration']

            manifest_lines.append({
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcription.strip()
            })
        except KeyError as e:
            print(f"Skipping entry {key} due to missing key: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    print(f"Successfully created raw NeMo manifest at: {output_path}")

# # --- Part 2: Split Manifest into Training and Validation Sets ---
# def split_manifest(config):
#     """Splits a manifest file into training and validation sets."""
#     input_path = config['raw_nemo_manifest_path']
#     train_out_path = config['train_split_path']
#     val_out_path = config['val_split_path']
#     val_ratio = config.get('val_ratio', 0.1)
#     seed = config.get('seed', 42)
    
#     print(f"\nStep 2: Splitting {input_path} into training and validation sets...")
#     with open(input_path, 'r', encoding='utf-8') as f:
#         lines = f.readlines()

#     random.seed(seed)
#     random.shuffle(lines)

#     val_size = int(len(lines) * val_ratio)
#     val_lines = lines[:val_size]
#     train_lines = lines[val_size:]

#     with open(train_out_path, 'w', encoding='utf-8') as f_train, \
#          open(val_out_path, 'w', encoding='utf-8') as f_val:
#         f_train.writelines(train_lines)
#         f_val.writelines(val_lines)
        
#     print(f"Split completed: {len(train_lines)} train samples, {len(val_lines)} validation samples.")
#     print(f"Train split manifest at: {train_out_path}")
#     print(f"Validation split manifest at: {val_out_path}")

# --- Part 3: Apply Kinyarwanda-Specific Text Normalization ---
def normalize_kinyarwanda_manifest(config):
    """Applies language-specific normalization to manifest files."""
    
    # Helper functions for text processing
    def remove_special_characters(data):
        chars_to_ignore_regex = r"[\.\,\?\:\-!;()«»…\]\[/\\*–‽+&_\\½√>€™$•¼}{~—=“\"”″‟„]"
        apostrophes_regex = r"[’'‘`ʽ']"
        text = data["text"].lower() # Convert to lowercase first
        text = re.sub(chars_to_ignore_regex, " ", text)
        text = re.sub(apostrophes_regex, "'", text)
        text = re.sub(r"'+", "'", text)
        text = re.sub(r"([b-df-hj-np-tv-z])' ([aeiou])", r"\1'\2", text)
        text = re.sub(r" '", " ", text)
        text = re.sub(r"' ", " ", text)
        data["text"] = re.sub(r" +", " ", text).strip()
        return data

    def replace_diacritics(data):
        text = data["text"]
        text = re.sub(r"[éèëēê]", "e", text)
        text = re.sub(r"[ãâāá]", "a", text)
        text = re.sub(r"[úūü]", "u", text)
        text = re.sub(r"[ôōó]", "o", text)
        text = re.sub(r"[ćç]", "c", text)
        text = re.sub(r"[ķ]", "k", text)
        text = re.sub(r"[ïī]", "i", text)
        text = re.sub(r"[ñ]", "n", text)
        data["text"] = text
        return data

    def remove_oov_characters(data):
        oov_regex = r"[^ 'aiuenrbomkygwthszdcjfvpl]"
        data["text"] = re.sub(oov_regex, "", data["text"]).strip()
        return data

    def process_manifest(input_path, output_path):
        manifest_data = read_manifest(input_path)
        preprocessors = [remove_special_characters, replace_diacritics, remove_oov_characters]
        
        for processor in preprocessors:
            for idx in tqdm(range(len(manifest_data)), desc=f"Applying {processor.__name__} to {os.path.basename(input_path)}"):
                manifest_data[idx] = processor(manifest_data[idx])
        
        processed_data = [entry for entry in manifest_data if entry['text'].strip()]
        write_manifest(output_path, processed_data)
        print(f"Finished writing normalized manifest: {output_path}")

    print("\nStep 3: Applying Kinyarwanda-specific normalization...")
    # Process the training split
    process_manifest(config['input_path'], config['processed_path'])


# --- Main Execution Block ---
if __name__ == '__main__':
    # Define all paths and parameters in a configuration dictionary
    pipeline_config = {
        # Part 1 Inputs
        'initial_json_path': '/shared/A_track/train.json',
        'audio_base_path': '/shared/track_a_audio_files',
        
        # Part 1 Output / Part 2 Input
        'raw_nemo_manifest_path': '/shared/A_track/train_nemo.json',
        
        # Part 2 Outputs / Part 3 Inputs
        'input_path': '/shared/A_track/train_nemo.json',
        # 'val_split_path': '/shared/A_track/val_split.json',
        
        # Part 3 Outputs (Final files for training)
        'processed_path': '/shared/A_track/train_processed.json',
        # 'val_processed_path': '/shared/A_track/val_processed.json',

    }

    eval_pipeline_config = {
        # Part 1 Inputs
        'initial_json_path': '/shared/A_track/dev_test.json',
        'audio_base_path': '/shared/track_a_audio_files',
        
        # Part 1 Output / Part 2 Input
        'raw_nemo_manifest_path': '/shared/A_track/val_nemo.json',
        
        # Part 2 Outputs / Part 3 Inputs
        'input_path': '/shared/A_track/val_nemo.json',
        # 'val_split_path': '/shared/A_track/val_split.json',
        
        # Part 3 Outputs (Final files for training)
        'processed_path': '/shared/A_track/val_processed.json',
        # 'val_processed_path': '/shared/A_track/val_processed.json',

    }

    # # Run the full pipeline
    convert_to_nemo_manifest(pipeline_config)
    normalize_kinyarwanda_manifest(pipeline_config)

    convert_to_nemo_manifest(eval_pipeline_config)
    normalize_kinyarwanda_manifest(eval_pipeline_config)
