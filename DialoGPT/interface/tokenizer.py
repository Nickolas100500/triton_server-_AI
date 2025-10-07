import numpy as np
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# Встановлюємо pad_token якщо його немає
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def encode_with_padding(prompt, target_length=128):
    # Токенізуємо без padding спочатку
    tokens = tokenizer(prompt, return_tensors="np", add_special_tokens=False)
    input_ids = tokens["input_ids"].astype(np.int32)

    current_length = input_ids.shape[1]
    
    # Використовуємо EOS як pad token
    pad_token_id = tokenizer.eos_token_id  # 50256 для DialoGPT
    
    if current_length < target_length:
        # Padding додаємо СПРАВА (в кінець)
        padding = np.full((1, target_length - current_length), pad_token_id, dtype=np.int32)
        input_ids = np.concatenate([input_ids, padding], axis=1)
    elif current_length > target_length:
        input_ids = input_ids[:, :target_length]
    
    return input_ids
