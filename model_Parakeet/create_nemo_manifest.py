import pandas as pd
import json
import os
from tqdm import tqdm

def convert_to_nemo_manifest(input_path, output_path, audio_base_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    manifest_lines = []
    for key, entry in tqdm(data.items(), total=len(data)):
        audio_file = entry['audio_path'] + '.wav'
        audio_path = os.path.join(audio_base_path, audio_file)
        try:
            audio_file = entry['audio_path'] + '.wav'
            audio_path = os.path.join(audio_base_path, audio_file)
            transcription = entry['transcription']
            duration = entry['duration']

            manifest_lines.append({
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcription.strip()
            })
        except KeyError as e:
            print(f"Skipping {key} due to missing key: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        for line in manifest_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

# convert_to_nemo_manifest('data/train.json', 'data/train_nemo.json', 'data/audio')


convert_to_nemo_manifest('shared/A_track/train.json', 
                         'shared/A_track/train_nemo.json',
                          'shared/track_a_audio_files')