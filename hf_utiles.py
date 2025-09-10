

def load_and_prepare_batch(batch):
    """
    This function is applied on-the-fly to a batch of data.
    """
    # Load the pre-computed Mel spectrograms from disk
    # The batch["file_path"] contains the relative paths from your CSV
    # --- THIS IS THE LINE TO CHANGE ---
    mel_tensors = [torch.load(raw_path + path, weights_only=False) for path in batch["file_path"]]

    # Tokenize the transcriptions
    labels = processor.tokenizer(batch["transcription"], padding="longest", return_tensors="pt").input_ids

    # The transformation function must return a dictionary
    # with keys that the model expects: 'input_features' and 'labels'.
    return {"input_features": mel_tensors, "labels": labels}

# --- IMPORTANT ---
# You must apply this new function again using set_transform
# before starting the trainer.

# dataset.set_transform(load_and_prepare_batch)

import torch

def data_collator_with_transformation(features):
    """
    This "all-in-one" data collator performs both the transformation (loading/tokenizing)
    and the padding.

    Args:
        features (list): A list of raw dataset examples, e.g.,
                         [{'file_path': '...', 'transcription': '...'}, ...]
    """
    # =================================================================
    # Part 1: Transformation (The work previously done by set_transform)
    # =================================================================
    # We loop through each raw example in the features list
    transformed_input_features = []
    transformed_label_features = []
    
    for feature in features:
        # Load the audio tensor from disk for the current example
        audio_tensor = torch.load(raw_path + feature["file_path"], weights_only=False)
        transformed_input_features.append({"input_features": audio_tensor})
        
        # Tokenize the transcription for the current example
        tokenized_label = processor.tokenizer(feature["transcription"]).input_ids
        transformed_label_features.append({"input_ids": tokenized_label})

    # =================================================================
    # Part 2: Padding (The work previously done by the old collator)
    # =================================================================
    # Pad the audio spectrograms
    batch = processor.feature_extractor.pad(transformed_input_features, return_tensors="pt")
    
    # Pad the text labels
    labels_batch = processor.tokenizer.pad(transformed_label_features, return_tensors="pt")

    # Replace the tokenizer's pad_token_id with -100 for the loss function
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    # If the bos token is appended in the preprocessing, we need to remove it
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    batch["labels"] = labels
    
    return batch

import torch
import warnings
from pickle import UnpicklingError  # <-- THE FIX: Add this import

def robust_data_collator(features):
    """
    This is the fully corrected robust data collator. It now imports UnpicklingError
    so it can properly catch all potential file loading errors.
    """
    valid_features = []
    
    # Part 1: Transformation (with error handling)
    for feature in features:
        try:
            # Attempt to load the audio tensor from disk
            audio_tensor = torch.load(raw_path + feature["file_path"], weights_only=False)
            
            # If successful, tokenize the transcription and add to our list
            tokenized_label = processor.tokenizer(feature["transcription"]).input_ids
            valid_features.append({
                "input_features": audio_tensor,
                "labels": tokenized_label
            })
            
        # This except block will now work correctly
        except (EOFError, RuntimeError, UnpicklingError) as e:
            # If torch.load fails, it's a corrupted file.
            warnings.warn(f"Skipping corrupted file: {feature['file_path']} | Error: {e}")
            continue

    # Part 2: Padding (only on the valid features)
    if not valid_features:
        warnings.warn("An entire batch of data was corrupted and has been skipped.")
        return {}
        
    input_features_list = [{"input_features": feature["input_features"]} for feature in valid_features]
    label_features_list = [{"input_ids": feature["labels"]} for feature in valid_features]

    batch = processor.feature_extractor.pad(input_features_list, return_tensors="pt")
    labels_batch = processor.tokenizer.pad(label_features_list, return_tensors="pt")
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    batch["labels"] = labels
    
    return batch




# ===================================================================================
# 1. DEFINE YOUR OWN CUSTOM DATA COLLATOR FUNCTION
# This function replaces DataCollatorSpeechSeq2SeqWithPadding entirely.
# ===================================================================================
def custom_data_collator(features):
    """
    A custom data collator that manually pads the input features and labels.
    
    Args:
        features (list): A list of feature dictionaries from the dataset.
    """
    print(features[0]["transcription"])
    # Isolate the input features and labels from the list of dictionaries
    input_features = [{"input_features": feature["input_features"]} for feature in features]
    label_features = [{"input_ids": feature["labels"]} for feature in features]
    # print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


    # Use the processor's feature_extractor to pad the audio spectrograms
    batch = processor.feature_extractor.pad(input_features, return_tensors="pt")
    
    # Use the processor's tokenizer to pad the text labels
    labels_batch = processor.tokenizer.pad(label_features, return_tensors="pt")

    # The model expects the padding in the labels to be -100, so we replace the
    # tokenizer's pad_token_id with -100.
    labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

    # If the bos token is appended in the preprocessing, we need to remove it
    if (labels[:, 0] == processor.tokenizer.bos_token_id).all().cpu().item():
        labels = labels[:, 1:]

    batch["labels"] = labels
    
    return batch



import torch

def adapt_model_to_80_bins(model):
    """
    This CORRECTED function surgically replaces the first convolutional layer
    of the Whisper model using the proper get_encoder() method.
    """
    print("Adapting the model to accept 80-channel input...")
    
    # --- THIS IS THE CORRECTED PART ---
    # Get the encoder using the correct method
    whisper_encoder = model.get_encoder()
    
    # Target the first convolutional layer from the encoder
    original_conv1 = whisper_encoder.conv1
    # --- END OF CORRECTION ---
    
    # Create a new layer with the same parameters but a different `in_channels`
    new_conv1 = torch.nn.Conv1d(
        in_channels=80,  # <-- The new, correct number of input channels
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=(original_conv1.bias is not None)
    )
    
    # Copy the weights from the original layer for the first 80 channels
    new_conv1.weight.data = original_conv1.weight.data[:, :80, :]
    if original_conv1.bias is not None:
        new_conv1.bias.data = original_conv1.bias.data
        
    # Replace the old layer with our new, adapted layer IN THE CORRECT LOCATION
    whisper_encoder.conv1 = new_conv1
    
    print("✅ Model successfully adapted.")
    return model



# This list will store predictions and labels for a few steps to help you visually inspect them
# It's for debugging and not required for the training itself.
all_predictions = []

import evaluate
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    # --- THIS IS THE DEBUGGING PRINT STATEMENT ---
    print("\n✅ --- Successfully called compute_metrics function! --- ✅\n")
    
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True, normalize=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True, normalize=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)


    print(f"Computed WER: {wer}") # Also good to print the result

    return {"wer": wer}

