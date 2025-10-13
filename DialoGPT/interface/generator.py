import numpy as np
import sys
import os

# Додаємо шлях до DialoGPT директорії
dialogpt_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if dialogpt_path not in sys.path:
    sys.path.insert(0, dialogpt_path)

from interface.client import TritonClient  # виправлено: імпортуємо клас
try:
    # Optional: use structured fallbacks if available
    from utils.fallbacks import DialoGPTFallbacks  # type: ignore
except Exception:
    DialoGPTFallbacks = None  # type: ignore
    
from interface.tokenizer import tokenizer, encode_with_padding

# Імпортуємо налаштування з DialoGPT/config.py
from config import (
    TEMPERATURE,
    TOP_K,
    MAX_NEW_TOKENS,
    TARGET_LENGTH,
    MIN_TOKENS_BEFORE_EOS,
)

def get_triton_client():
    # Створюємо екземпляр клієнта Triton
    return TritonClient()

class DialoGPTGenerator:
    """Клас-обгортка для генерації відповідей через Triton із fallback.

    Зберігає доступ до конфігів та надає методи generate_response і get_fallback_response.
    """

    def __init__(self, fallbacks_provider=None):
        self.temperature = TEMPERATURE
        self.top_k = TOP_K
        self.max_new_tokens_default = MAX_NEW_TOKENS
        self.target_length_default = TARGET_LENGTH
        self.min_tokens_before_eos = MIN_TOKENS_BEFORE_EOS
        self._fallbacks = fallbacks_provider

    def get_client(self):
        return get_triton_client()

    def run_inference(self, client, model_name, input_ids):
        # Викликаємо метод класу TritonClient
        return client.run_inference(model_name, input_ids)


    def generate_response(self, prompt, model_name="dialogpt_onnx", target_length=None, max_new_tokens=None):
        target_length = target_length or self.target_length_default
        max_new_tokens = max_new_tokens or self.max_new_tokens_default

        try:
            client = self.get_client()
            print("🔍 Спроба генерації через Triton сервер (1 запит)...")
            print(f"⚙️  Налаштування: temp={self.temperature}, top_k={self.top_k}, max_tokens={max_new_tokens}")

            conversation_prompt = f"Human: {prompt}\nBot:"
            input_ids = encode_with_padding(conversation_prompt, target_length)

            # Передаємо одразу max_new_tokens (якщо Triton-модель це підтримує)
            logits = self.run_inference(client, model_name, input_ids)

            # Постобробка: генеруємо відповідь на основі logits (sampling/argmax)

            generated_token_ids = []
            pad_token_id = tokenizer.eos_token_id
            current_pos = np.where(input_ids[0] != pad_token_id)[0]
            if len(current_pos) > 0:
                start_pos = current_pos[-1]
            else:
                start_pos = target_length - max_new_tokens

            for step in range(max_new_tokens):
                idx = start_pos + step
                if idx >= logits.shape[1]:
                    break
                step_logits = logits[0, idx, :].copy()
                if step < self.min_tokens_before_eos:
                    step_logits[tokenizer.eos_token_id] = -1000
                step_logits = step_logits / self.temperature
                k = self.top_k
                top_k_indices = np.argpartition(step_logits, -k)[-k:]
                top_k_logits = step_logits[top_k_indices]
                top_k_logits = top_k_logits - np.max(top_k_logits)
                top_k_probs = np.exp(top_k_logits)
                top_k_probs = top_k_probs / np.sum(top_k_probs)
                sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
                next_token_id = top_k_indices[sampled_idx]
                if next_token_id == tokenizer.eos_token_id:
                    if len(generated_token_ids) >= self.min_tokens_before_eos:
                        break
                generated_token_ids.append(next_token_id)

            response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

            if len(response) < 3:
                print(f"⚠️  TRITON генерація не успішна (коротка відповідь: '{response}') - використовую FALLBACK")
                return self.get_fallback_response(prompt)

            print(f"✅ TRITON УСПІШНО: '{response}' ({len(generated_token_ids)} токенів)")
            return response

        except Exception as e:
            print(f"❌ Помилка Triton сервера: {e}")
            print("🔄 Переходжу на FALLBACK відповідь...")
            return self.get_fallback_response(prompt)
        
    def generate_response(self, prompt, model_name="dialogpt_onnx", target_length=None, max_new_tokens=None):
            target_length = target_length or self.target_length_default
            max_new_tokens = max_new_tokens or self.max_new_tokens_default

            try:
                client = self.get_client()
                print("🔍 Спроба генерації через Triton сервер (autoregressive)...")
                print(f"⚙️  Налаштування: temp={self.temperature}, top_k={self.top_k}, max_tokens={max_new_tokens}")

                conversation_prompt = f"Human: {prompt}\nBot:"
                input_ids = encode_with_padding(conversation_prompt, target_length)

                generated_token_ids = []
                pad_token_id = tokenizer.eos_token_id
                for step in range(max_new_tokens):
                    logits = self.run_inference(client, model_name, input_ids)
                    non_pad_positions = np.where(input_ids[0] != pad_token_id)[0]
                    if len(non_pad_positions) > 0:
                        current_pos = non_pad_positions[-1]
                    else:
                        print("⚠️ No non-pad tokens found!")
                        break
                    last_token_logits = logits[0, current_pos, :].copy()
                    if step < self.min_tokens_before_eos:
                        last_token_logits[tokenizer.eos_token_id] = -1000
                    last_token_logits = last_token_logits / self.temperature
                    k = self.top_k
                    top_k_indices = np.argpartition(last_token_logits, -k)[-k:]
                    top_k_logits = last_token_logits[top_k_indices]
                    top_k_logits = top_k_logits - np.max(top_k_logits)
                    top_k_probs = np.exp(top_k_logits)
                    top_k_probs = top_k_probs / np.sum(top_k_probs)
                    sampled_idx = np.random.choice(len(top_k_probs), p=top_k_probs)
                    next_token_id = top_k_indices[sampled_idx]
                    if next_token_id == tokenizer.eos_token_id:
                        if len(generated_token_ids) >= self.min_tokens_before_eos:
                            break
                    generated_token_ids.append(next_token_id)
                    # Вставляємо новий токен у перший паддінг
                    first_pad_pos = np.where(input_ids[0] == pad_token_id)[0]
                    if len(first_pad_pos) > 0:
                        input_ids[0, first_pad_pos[0]] = next_token_id
                    else:
                        # Якщо паддінгу немає — перекодуємо весь prompt
                        decoded_so_far = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True) + tokenizer.decode([next_token_id], skip_special_tokens=True)
                        input_ids = encode_with_padding(decoded_so_far, target_length)

                response = tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()

                if len(response) < 3:
                    print(f"⚠️  TRITON генерація не успішна (коротка відповідь: '{response}') - використовую FALLBACK")
                    return self.get_fallback_response(prompt)

                print(f"✅ TRITON УСПІШНО: '{response}' ({len(generated_token_ids)} токенів)")
                return response

            except Exception as e:
                print(f"❌ Помилка Triton сервера: {e}")
                print("🔄 Переходжу на FALLBACK відповідь...")
                return self.get_fallback_response(prompt)

    def get_fallback_response(self, prompt):
        if self._fallbacks and hasattr(self._fallbacks, 'get_response'):
            try:
                text = self._fallbacks.get_response(prompt)
                if isinstance(text, str) and text:
                    print(f"🎯 FALLBACK(provider): '{text}'")
                    return text
            except Exception as _e:
                pass

        prompt_lower = prompt.lower()

        match True:
            case _ if any(word in prompt_lower for word in ("hello", "hi", "hey")):
                fallback_response = "Hello! How can I help you today?"
            case _ if any(word in prompt_lower for word in ("how are you", "how do you do")):
                fallback_response = "I'm doing well, thank you for asking!"
            case _ if any(word in prompt_lower for word in ("name", "who are you")):
                fallback_response = "I'm DialoGPT, an AI assistant. What's your name?"
            case _ if any(word in prompt_lower for word in ("joke", "funny")):
                fallback_response = "Here's a joke: Why don't programmers like nature? It has too many bugs!"
            case _ if any(word in prompt_lower for word in ("help", "assist")):
                fallback_response = "I'd be happy to help! What do you need assistance with?"
            case _:
                fallback_response = "That's interesting! Could you tell me more about that?"

        print(f"🎯 FALLBACK: '{fallback_response}'")
        return fallback_response

# --- Backward-compatible functional API ---
_DEFAULT_GENERATOR = DialoGPTGenerator()

def generate_response(prompt, model_name="dialogpt_onnx", target_length=None, max_new_tokens=None):
    return _DEFAULT_GENERATOR.generate_response(prompt, model_name, target_length, max_new_tokens)

def get_fallback_response(prompt):
    return _DEFAULT_GENERATOR.get_fallback_response(prompt)