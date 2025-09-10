import torch
import os
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


config = {
    'model_config_path': 'examples/asr/conf/conformer/conformer_ctc_bpe.yaml', # Path inside NeMo repo
    'model': {
        'init_from_pretrained_model': 'nvidia/parakeet-ctc-1.1b',
        'tokenizer_dir': './kinyarwanda_tokenizers/tokenizer_spe_bpe_v1024',
        'train_ds': {
            'manifest_filepath': 'shared/A_track/train_processed.json',
            'batch_size': 16,
             'max_duration': 30.0 # Increased max duration
        },
        'validation_ds': {
            'manifest_filepath': '/shared/A_track/val_processed.json',
            'batch_size': 16,
            'max_duration': 30.0 # Increased max duration
        },
        'optim': {
            'name': 'adamw',
            'lr': 0.0001,
            'betas': [0.9, 0.98],
            'weight_decay': 0.001,
            'sched': {
                'name': 'CosineAnnealing',
                'warmup_steps': 2000
            }
        }
    },
    'trainer': {
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1, 
        'max_epochs': 50,
        'precision': 'bf16'
    },


        'exp_manager': {
            'exp_dir': '/shared/KASR/nemo/nemo_experiments',
            'create_wandb_logger': True,
            'wandb_logger_kwargs': {
                'name': 'parakeet-kinyarwanda-two-phase-finetune',
                'project': 'nemo-asr',
    'resume': 'must',
    'id': "2025-06-20_19-52-16"

                                    },
            'create_checkpoint_callback': True,
            'checkpoint_callback_params': {
                'monitor': 'val_wer',  # The metric to monitor
                'mode': 'min',         # 'min' for error rates, 'max' for accuracy
                'save_top_k': 5,       # Save the top 5 models
                'filename': '{epoch}-{step}-{val_wer:.2f}', # Name checkpoints with their WER
                'verbose': True,
                                        },

                 'resume_if_exists': True,
                    'resume_ignore_no_checkpoint': True
                    }
    }







# programmatic_finetuning.py
import os
# import pytorch_lightning as pl
import lightning.pytorch as pl 
from omegaconf import OmegaConf
import nemo.collections.asr as nemo_asr
from nemo.utils.exp_manager import exp_manager


"""
Launches a NeMo ASR fine-tuning job programmatically.

Args:
    config (dict): A dictionary containing all necessary configuration parameters.
"""
print("--- Starting Programmatic Fine-Tuning ---")

# --- 1. Set up PyTorch Lightning Trainer ---
# The trainer is responsible for managing the training loop.
trainer_config = config['trainer']
trainer = pl.Trainer(**trainer_config, logger=False, enable_checkpointing=False)

# --- 2. Set up Experiment Manager ---
# The experiment manager handles logging, checkpointing, and experiment organization.
exp_manager_config = config.get('exp_manager', {})
# The `exp_manager` function requires the trainer to be passed to it.
exp_dir = exp_manager(trainer, exp_manager_config)
# 


# --- 3. Load Pretrained Model ---
print(f"Loading pretrained model: {config['model']['init_from_pretrained_model']}")
asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(
    model_name=config['model']['init_from_pretrained_model'],
    trainer=trainer,
    
)

asr_model.cfg.train_ds.batch_size =8




# --- 4. Update Model Configuration ---
# print("Updating model configuration for Kinyarwanda fine-tuning...")
# with open(config['model_config_path'], 'r') as f:
#     model_cfg = OmegaConf.load(f)
model_cfg = asr_model.cfg

model_cfg.train_ds.batch_size = 6
model_cfg.validation_ds.batch_size = 6

model_cfg.train_ds.max_duration = 30


# Override tokenizer and dataset paths
model_cfg.tokenizer.dir = config['model']['tokenizer_dir']
model_cfg.train_ds.manifest_filepath = config['model']['train_ds']['manifest_filepath']
model_cfg.validation_ds.manifest_filepath = config['model']['validation_ds']['manifest_filepath']


# Set up the new tokenizer and vocabulary for the model
asr_model.change_vocabulary(new_tokenizer_dir=model_cfg.tokenizer.dir, new_tokenizer_type='bpe')

# model_cfg.max_duration = 30

# Set up the data loaders with the new configuration
asr_model.setup_training_data(model_cfg.train_ds)
asr_model.setup_validation_data(model_cfg.validation_ds)



model_cfg.optim = OmegaConf.create(config['model']['optim'])
# Override optimizer and scheduler parameters
# OmegaConf.update(model_cfg.optim, config['model']['optim'], merge=True)
asr_model.setup_optimization(optim_config=model_cfg.optim)


# --- 5. Start Fine-Tuning ---
print("Configuration complete. Starting training...")
# trainer.fit(asr_model)
asr_model.train()
trainer.fit(asr_model, ckpt_path="/shared/KASR/nemo/nemo_experiments/default/2025-06-20_19-52-16/checkpoints/epoch=1-step=26888-val_wer=0.19-last.ckpt")
print("Fine-tuning complete.")

# --- 6. Save the Final Model ---
final_model_path = os.path.join(exp_dir, "finetuned_kinyarwanda_model.nemo")
asr_model.save_to(final_model_path)
print(f"Final fine-tuned model saved to: {final_model_path}")



