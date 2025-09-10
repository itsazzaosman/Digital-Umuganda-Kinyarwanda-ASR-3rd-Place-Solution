from pytorch_lightning import Trainer
import os

os.environ["NEMO_CACHE_DIR"] = "shared/A_track/nemo_cache"

import nemo.collections.asr as nemo_asr

model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-1.1b")

# Setup subset data
model.setup_training_data(train_data_config={
    "sample_rate": 16000,
    "manifest_filepath": "shared/A_track/train_split_nemo.json",
    "batch_size": 2,
    "shuffle": True
})
model.setup_validation_data(val_data_config={
    "sample_rate": 16000,
    "manifest_filepath": "shared/A_track/val_split_nemo.json",
    "batch_size": 2,
    "shuffle": False
})

model._setup_tokenizer(
    train_ds=model._train_dl.dataset,
    update_tokenizer=True,
    tokenizer_type="bpe",
    tokenizer_dir="shared/A_track/tokenizer_out_dir"
)


trainer = Trainer(
    accelerator="cuda", 
    devices=1,
    max_epochs=1,
    limit_train_batches=5,
    limit_val_batches=2,
    log_every_n_steps=1
)

trainer.fit(model)

print("Dry run completed on CPU.")