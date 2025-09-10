python /shared/KASR/nemo/scripts/process_asr_text_tokenizer.py \
       --manifest="/shared/A_track/train_processed.json" \
       --data_root="./kinyarwanda_tokenizers" \
       --vocab_size=1024 \
       --tokenizer="spe" \
       --spe_type="bpe" \
        --spe_remove_extra_whitespaces \
       --log