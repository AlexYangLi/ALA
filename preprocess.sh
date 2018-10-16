cd preprocess

echo "preprocessing raw data..."
python process_raw.py

echo "generate word embeddings..."
python gen_word_embed.py

echo "done!"
