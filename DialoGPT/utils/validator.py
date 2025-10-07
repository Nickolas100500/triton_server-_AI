"""
Валідатор для DialoGPT системи.
Перевіряє Triton з'єднання, якість відповідей та інші аспекти системи.

Класова реалізація + сумісні зворотно функції-обгортки.
"""

import re
from tritonclient.grpc import InferenceServerClient, InferenceServerException


class DialoGPTValidator:
    """Клас-валідатор із конфігурацією та методами перевірок."""

    def __init__(
        self,
        triton_url: str = "127.0.0.1:8001",
        model_name: str = "dialogpt_onnx",
        prompt_max_len: int = 500,
        expected_logits_shape = (1, 128, 50257),
        dangerous_patterns = None,
    ) -> None:
        self.triton_url = triton_url
        self.model_name = model_name
        self.prompt_max_len = prompt_max_len
        self.expected_logits_shape = expected_logits_shape
        self.dangerous_patterns = dangerous_patterns or [
            r'<script',
            r'javascript:',
            r'eval\(',
            r'exec\(',
        ]

    def validate_triton_connection(self, url: str | None = None):
        """Перевіряє підключення до Triton сервера"""
        url = url or self.triton_url
        try:
            client = InferenceServerClient(url=url)

            # Перевіряємо статус сервера
            if not client.is_server_live():
                return {"valid": False, "error": "Server is not live"}

            if not client.is_server_ready():
                return {"valid": False, "error": "Server is not ready"}

            # Перевіряємо наявність моделі
            try:
                _ = client.get_model_metadata(self.model_name)
                return {
                    "valid": True,
                    "status": "Connected successfully",
                    "model": f"{self.model_name} available",
                }
            except Exception:
                return {"valid": False, "error": f"Model {self.model_name} not found"}

        except InferenceServerException as e:
            return {"valid": False, "error": f"Triton error: {e}"}
        except Exception as e:
            return {"valid": False, "error": f"Connection error: {e}"}

    def validate_response_quality(self, response: str):
        """Перевіряє якість згенерованої відповіді"""
        if not response or not isinstance(response, str):
            return {"valid": False, "issue": "Empty or invalid response"}

        response = response.strip()

        # Мінімальна довжина
        if len(response) < 2:
            return {"valid": False, "issue": "Response too short"}

        # Перевірка на повторення символів
        if len(set(response.replace(" ", ""))) <= 2 and len(response) > 5:
            return {"valid": False, "issue": "Repetitive characters"}

        # Перевірка на тільки спецсимволи
        if re.match(r'^[^\w\s]+$', response):
            return {"valid": False, "issue": "Only special characters"}

        # Перевірка на тільки числа
        if re.match(r'^[\d\s]+$', response):
            return {"valid": False, "issue": "Only numbers"}

        # Добра якість
        word_count = len(response.split())
        has_letters = bool(re.search(r'[a-zA-Z]', response))

        return {
            "valid": True,
            "quality": "good" if word_count >= 3 and has_letters else "acceptable",
            "word_count": word_count,
            "has_letters": has_letters,
        }

    def validate_prompt(self, prompt: str):
        """Перевіряє валідність вхідного промпту"""
        print("🔍 validate_prompt called with:", prompt)
        if not prompt or not isinstance(prompt, str):
            return {"valid": False, "issue": "Empty or invalid prompt"}

        prompt = prompt.strip()

        if len(prompt) < 1:
            return {"valid": False, "issue": "Prompt too short"}

        if len(prompt) > self.prompt_max_len:
            return {"valid": False, "issue": f"Prompt too long (>{self.prompt_max_len} chars)"}

        # Перевірка на небезпечні символи
        for pattern in self.dangerous_patterns:
            if re.search(pattern, prompt, re.IGNORECASE):
                return {"valid": False, "issue": "Potentially dangerous content"}

        return {
            "valid": True,
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "issue": None,
        }

    def validate_model_output(self, logits):
        """Перевіряє output моделі"""
        import numpy as np

        if logits is None:
            return {"valid": False, "issue": "No logits returned"}

        if not isinstance(logits, np.ndarray):
            return {"valid": False, "issue": "Logits not numpy array"}

        expected_shape = self.expected_logits_shape
        if logits.shape != expected_shape:
            return {
                "valid": False,
                "issue": f"Wrong shape: {logits.shape}, expected: {expected_shape}",
            }

        # Перевірка на NaN або Inf
        if np.isnan(logits).any():
            return {"valid": False, "issue": "Contains NaN values"}

        if np.isinf(logits).any():
            return {"valid": False, "issue": "Contains Inf values"}

        return {
            "valid": True,
            "shape": logits.shape,
            "min_value": float(np.min(logits)),
            "max_value": float(np.max(logits)),
        }

    def run_full_validation(self):
        """Запускає повну валідацію системи"""
        print("🔍 ПОВНА ВАЛІДАЦІЯ DialoGPT СИСТЕМИ")
        print("=" * 50)

        results = {}

        # 1. Triton підключення
        print("1. Перевірка Triton сервера...")
        triton_result = self.validate_triton_connection()
        results["triton"] = triton_result
        if triton_result.get("valid"):
            print(f"   ✅ {triton_result.get('status')}")
        else:
            print(f"   ❌ {triton_result.get('error')}")

        # 2. Тест промптів
        print("\n2. Перевірка промптів...")
        test_prompts = ["Hello", "How are you?", "What's your name?", ""]
        for prompt in test_prompts:
            result = self.validate_prompt(prompt)
            display_prompt = f"'{prompt}'" if prompt else "'<empty>'"
            if result.get("valid"):
                print(f"   ✅ {display_prompt} - OK")
            else:
                print(f"   ❌ {display_prompt} - {result.get('issue')}")

        # 3. Тест якості відповідей
        print("\n3. Перевірка якості відповідей...")
        test_responses = [
            "Hello! How can I help you?",
            "!",
            "",
            "1 1 1 1 1",
            "abc abc abc",
        ]
        for response in test_responses:
            result = self.validate_response_quality(response)
            display_response = f"'{response}'" if response else "'<empty>'"
            if result.get("valid"):
                quality = result.get("quality", "unknown")
                print(f"   ✅ {display_response} - {quality}")
            else:
                print(f"   ❌ {display_response} - {result.get('issue')}")

        print("\n🏁 ВАЛІДАЦІЯ ЗАВЕРШЕНА")

        # Підсумок
        total_checks = sum(1 for r in results.values() if isinstance(r, dict) and "valid" in r)
        passed_checks = sum(1 for r in results.values() if isinstance(r, dict) and r.get("valid"))

        if passed_checks == total_checks and passed_checks > 0:
            print("🎉 Всі перевірки пройшли успішно!")
        else:
            print(f"⚠️  {passed_checks}/{total_checks} перевірок пройшли")

        return results


# --- Зворотно сумісні функції-обгортки ---
_DEFAULT_VALIDATOR = DialoGPTValidator()


def validate_triton_connection(url: str = "127.0.0.1:8001"):
    return _DEFAULT_VALIDATOR.validate_triton_connection(url)


def validate_response_quality(response: str):
    return _DEFAULT_VALIDATOR.validate_response_quality(response)


def validate_prompt(prompt: str):
    return _DEFAULT_VALIDATOR.validate_prompt(prompt)


def validate_model_output(logits):
    return _DEFAULT_VALIDATOR.validate_model_output(logits)


def run_full_validation():
    return _DEFAULT_VALIDATOR.run_full_validation()


if __name__ == "__main__":
    _DEFAULT_VALIDATOR.run_full_validation()
