#!/usr/bin/env python3
"""
DialoGPT FastAPI сервер з повною інтеграцією main.py логіки
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import sys

# Add DialoGPT path
sys.path.append(os.path.join(os.path.dirname(__file__), 'DialoGPT'))

# Ініціалізація FastAPI
app = FastAPI(
    title="DialoGPT Chat API",
    description="API для чату з DialoGPT ботом",
    version="1.0.0"
)

# Initialize templates with absolute path
template_dir = os.path.join(os.path.dirname(__file__), "template")
templates = Jinja2Templates(directory=template_dir)
print(f"✅ Templates loaded from: {template_dir}")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
"""
Attempt to import DialoGPT components. Keep API resilient if some pieces are missing.
"""
try:
    from DialoGPT.interface.generator import generate_response  # type: ignore
except Exception as e:
    print(f"⚠️ DialoGPT.generator unavailable: {e}")
    generate_response = None  # type: ignore

try:
    from DialoGPT.utils.validator import (  # type: ignore
        validate_response_quality,
        validate_triton_connection,
        validate_prompt as _validate_prompt_impl,
    )
except Exception as e:
    print(f"⚠️ DialoGPT.utils.validator unavailable: {e}")
    validate_response_quality = None  # type: ignore
    validate_triton_connection = None  # type: ignore
    _validate_prompt_impl = None  # type: ignore

DIALOGPT_AVAILABLE = generate_response is not None
if DIALOGPT_AVAILABLE:
    print("✅ DialoGPT components loaded (generator)")
else:
    print("⚠️ DialoGPT generator not available; will use fallback responses")

def safe_validate_prompt(prompt: str):
    """Local safe wrapper around validate_prompt to avoid NameError.

    Returns dict with at least {"valid": bool}.
    If validator is unavailable, treat non-empty input as valid.
    """
    try:
        if _validate_prompt_impl:
            return _validate_prompt_impl(prompt)
    except Exception as e:
        print(f"⚠️ validate_prompt error: {e}")
    # Fallback logic: basic checks
    if not isinstance(prompt, str) or not prompt.strip():
        return {"valid": False, "issue": "Empty or invalid prompt"}
    if len(prompt) > 500:
        return {"valid": False, "issue": "Prompt too long (>500 chars)"}
    return {"valid": True}

# Fallback responses for when DialoGPT is not available
FALLBACK_RESPONSES = [
    "Привіт! Як справи?",
    "Цікаве питання! Розкажіть більше.",
    "Я слухаю вас.",
    "Що вас цікавить?",
    "Давайте поговоримо про щось цікаве!",
    "Як ваші справи сьогодні?"
]

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    use_smart_fallback: bool = True

class ChatResponse(BaseModel):
    response: str
    source: str
    quality: bool
    intent: str = None


def generate_response_simple(prompt):
    """
    Спрощена генерація відповіді з інформацією про джерело.
    Повертає кортеж: (response: str, source: str, quality: bool, intent: str)
    """
    try:
        # Валідуємо промпт (safe wrapper)
        prompt_validation = safe_validate_prompt(prompt)
        if not prompt_validation['valid']:
            import random
            fb = random.choice(FALLBACK_RESPONSES)
            return fb, "fallback", True, "chat"

        # Якщо DialoGPT доступний, використовуємо його
        if DIALOGPT_AVAILABLE and generate_response:
            response = generate_response(prompt)

            # Перевіряємо якість відповіді
            response_validation = (
                validate_response_quality(response)
                if callable(validate_response_quality) else {"valid": True}
            )
            if response_validation.get('valid'):
                return response, "dialogpt", True, "chat"
            # Якщо відповідь неякісна — фолбек
            import random
            fb = random.choice(FALLBACK_RESPONSES)
            return fb, "fallback", True, "chat"

        # Якщо DialoGPT недоступний — фолбек
        import random
        fb = random.choice(FALLBACK_RESPONSES)
        return fb, "fallback", True, "chat"

    except Exception as e:
        print(f"❌ Error in generate_response_simple: {e}")
        return "Вибачте, сталася помилка. Спробуйте ще раз.", "error", False, "error"


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main chat page"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Serve the chat page (same as root)"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/health")
async def health_check():
    """Simple health/status endpoint for the UI"""
    return {
        "status": "ok",
        "dialogpt_chat": "available" if DIALOGPT_AVAILABLE else "unavailable"
    }

@app.post("/predict", response_model=ChatResponse)
async def chat_predict(request: ChatRequest):
    """DialoGPT chat endpoint"""
    
    try:
        message = (request.message or "").strip()
        if not message:
            raise ValueError("Empty message")
        
        # Генеруємо відповідь (+ джерело)
        response, source, quality, intent = generate_response_simple(message)

        print(f"✅ Chat response: '{response}' (source: {source})")

        return ChatResponse(
            response=response,
            source=source,
            quality=quality,
            intent=intent
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return ChatResponse(
            response="Вибачте, сталася технічна помилка.",
            source="error",
            quality=False,
            intent="error"
        )


# Add test call to generate_response for diagnostics
try:
    test_prompt = "Hello, how are you?"
    if callable(generate_response):
        test_response = generate_response(test_prompt, max_new_tokens=25)
        print(f"🔍 Test generate_response: {test_response}")
    else:
        print("ℹ️ Skipping test generate_response (generator unavailable)")
except Exception as e:
    print(f"❌ Test generate_response failed: {e}")


# Add optional diagnostics (controlled by CHAT_API_DEBUG env var)
import os
DEBUG = os.getenv("CHAT_API_DEBUG", "").lower() in {"1", "true", "yes", "on"}
if DEBUG:
    # Avoid printing full sys.path by default; keep minimal checks
    print("🔍 DialoGPT exists:", os.path.exists(os.path.join(os.path.dirname(__file__), 'DialoGPT')))
    print("🔍 utils/validator.py exists:", os.path.exists(os.path.join(os.path.dirname(__file__), 'DialoGPT', 'utils', 'validator.py')))

# Additional diagnostic to check validator availability (debug only)
if DEBUG:
    print(f"🔍 validator.impl available: {bool('_validate_prompt_impl' in globals() and _validate_prompt_impl)}")
    print(f"🔍 using safe_validate_prompt: {callable(safe_validate_prompt)}")

# To start the server, run:
# python -m uvicorn chat_api:app --reload --port 4000