"""
DialoGPT Configuration
"""

# Налаштування генерації
TEMPERATURE = 0.9           # Температура для sampling (0.1-2.0)
TOP_K = 50                  # Top-k sampling
MAX_NEW_TOKENS = 20         # Максимум нових токенів
TARGET_LENGTH = 128         # Довжина контексту

# Triton сервер
TRITON_URL = "127.0.0.1:8001"
MODEL_NAME = "dialogpt_onnx"

# Налаштування EOS
MIN_TOKENS_BEFORE_EOS = 3   # Мінімум токенів перед дозволом EOS