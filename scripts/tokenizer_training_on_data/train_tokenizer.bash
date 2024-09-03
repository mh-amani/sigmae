# scan:
python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type word \
    --dataset_path "scan" \
    --save_path "data/tokenizers/scan/actions" \
    --key "actions" \
    --batch_size 1000 \
    --max_vocab_size 1000 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"name": "simple"}'

python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type word \
    --dataset_path "scan" \
    --save_path "data/tokenizers/scan/commands" \
    --key "commands" \
    --batch_size 1000 \
    --max_vocab_size 1000 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"name": "simple"}'


# cogs:
python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type unigram \
    --dataset_path "data/cogs/train.tsv" \
    --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_input_5000" \
    --key "input" \
    --batch_size 1000 \
    --max_vocab_size 5000 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"delimiter": "\t", "column_names": ["input", "output", "in_dist_or_out"]}'

python scripts/tokenizer_training_on_data/train_tokenizer_on_dataset.py \
    --tokenizer_type unigram \
    --dataset_path "data/cogs/train.tsv" \
    --save_path "data/tokenizers/cogs/cogs_unigram_tokenizer_output_5000" \
    --key "output" \
    --batch_size 1000 \
    --max_vocab_size 5000 \
    --special_tokens "[PAD]" "[UNK]" "[BOS]" "[EOS]" \
    --dataset_config '{"delimiter": "\t", "column_names": ["input", "output", "in_dist_or_out"]}'