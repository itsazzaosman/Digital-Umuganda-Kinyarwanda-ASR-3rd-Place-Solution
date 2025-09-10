# programmatic_finetuning.py
import os
import torch
# import pytorch_lightning as pl
import lightning.pytorch as pl 
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager

import utils

# import KASR.nemo.utils as utils
device = utils.get_device_safe_threading()

# for h100
torch.set_float32_matmul_precision('high') # medium

config = {
    'model': {
        'tokenizer_dir': 'shared/KASR/nemo/kinyarwanda_tokenizers/tokenizer_spe_bpe_v1024',
        'train_ds': {
            'manifest_filepath': 'shared/A_track/train_processed.json',
            'batch_size': 24,
            'shuffle': True,
            'num_workers': 12,
             'max_duration': 30.0 # Increased max duration
        },
        'validation_ds': {
            'manifest_filepath': 'shared/A_track/val_processed.json',
            'batch_size': 24,
        },
        'optim': {
            'name': 'adamw',
            'lr': 0.00001,
            'betas': [0.9, 0.98],
            'weight_decay': 0.0005,
            'sched': {
                'name': 'CosineAnnealing',
                'warmup_steps': 500,
            },
       
        }
    },
    'trainer': {
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1, 
        'max_epochs': 20,
        'precision': 'bf16'
    },     
    
}


# --- 1. Set up PyTorch Lightning Trainer ---
# The trainer is responsible for managing the training loop.
trainer_config = config['trainer']
trainer = pl.Trainer(**trainer_config,logger=False,  enable_checkpointing=False)

# --- 2. Set up Experiment Manager ---
# The experiment manager handles logging, checkpointing, and experiment organization.
exp_manager_config = {
            'exp_dir': 'shared/KASR/nemo/nemo_experiments',
            'create_wandb_logger': True,
            'wandb_logger_kwargs': {
                'name': 'parakeet-kinyarwanda-two-phase-finetune',
                'project': 'nemo-asr',
    'resume': 'must',
    'id': "2025-06-27_05-19-09"


                                    },
            'create_checkpoint_callback': True,
            'checkpoint_callback_params': {
                'monitor': 'val_wer',  # The metric to monitor
                'mode': 'min',         # 'min' for error rates, 'max' for accuracy
                'save_top_k': 5,       # Save the top 5 models
                'filename': '{epoch}-{step}-{val_wer:.4f}', # Name checkpoints with their WER
                'verbose': True,
                                        },
                 'resume_if_exists': True,
                'resume_ignore_no_checkpoint': True, ## if true, it will check the files 
                    }

# ckpt_path = None
ckpt_path= "shared/KASR/nemo/nemo_experiments/default/checkpoints/epoch=5-step=22410-val_wer=0.0887.ckpt"
#"shared/epoch=50-step=168050-val_wer=0.09-last.ckpt"

restore_path = "shared/KASR_2/nemo/nemo_experiments/default/2025-06-22_02-42-58/checkpoints/default.nemo"

exp_dir = exp_manager(trainer, exp_manager_config)



# --- 3. Load Pretrained Model ---
print(f"Loading pretrained model: 'nvidia/parakeet-ctc-1.1b'")
# asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
#     model_name='nvidia/parakeet-ctc-1.1b',
#     trainer=trainer,
    
# )


# Load model from .ckpt
asr_model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(ckpt_path, 
                                                                   map_location="cuda" if torch.cuda.is_available() else "cpu")

asr_model.trainer = trainer

# asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(restore_path=restore_path,
#                                                            trainer=trainer,
#                                                            map_location="cuda",)


# --- 4. Update Model Configuration ---
# print("Updating model configuration for Kinyarwanda fine-tuning...")



model_cfg = asr_model.cfg

# Override tokenizer and dataset paths
# model_cfg.tokenizer.dir = config['model']['tokenizer_dir']

# asr_model.change_vocabulary(new_tokenizer_dir=model_cfg.tokenizer.dir, new_tokenizer_type='bpe')

# Set up the data loaders with the new configuration
for k, v in config['model']['train_ds'].items():
    OmegaConf.update(model_cfg.train_ds, k, v)

asr_model.setup_training_data(model_cfg.train_ds)

for k, v in config['model']['validation_ds'].items():
    OmegaConf.update(model_cfg.validation_ds, k, v)

asr_model.setup_validation_data(model_cfg.validation_ds)


# # Override optimizer and scheduler parameters
for k, v in config['model']['optim'].items():
    OmegaConf.update(model_cfg.optim, k, v)


# # if not ckpt_path:
asr_model.setup_optimization(optim_config=model_cfg.optim)

# Set the model to use greedy decoding strategy during inference
# asr_model.decoding.strategy = 'greedy_batch'


# --- 5. Start Fine-Tuning ---
print("Configuration complete. Starting training...")
# trainer.fit(asr_model)
asr_model.train()
trainer.fit(asr_model)
print("Fine-tuning complete.")

# --- 6. Save the Final Model ---
final_model_path = os.path.join(exp_dir, "finetuned_kinyarwanda_model.nemo")
asr_model.save_to(final_model_path)
print(f"Final fine-tuned model saved to: {final_model_path}")



