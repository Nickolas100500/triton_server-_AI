# DialoGPT Modular System

Модульна система для запуску DialoGPT з Triton Inference Server та fallback механізмами.

## Структура проекту

```
DialoGPT/
├── main.py              # Головний файл додатку
├── config.py            # Централізована конфігурація
├── fallbacks.json       # JSON дані для fallback відповідей
├── core/                # Основні модулі інференсу
│   ├── __init__.py
│   ├── inferens.py      # Функції інференсу Triton
│   ├── tokenization.py  # Токенізація та обробка тексту
│   └── validation.py    # Валідація моделей, відповідей та промптів
├── utils/               # Утилітарні модулі
│   ├── __init__.py
│   └── data_loader.py   # Завантаження JSON даних
├── generation/          # Генерація відповідей
│   ├── __init__.py
│   ├── response_generator.py  # Генерація через Triton/fallback
│   ├── token_filtering.py     # Фільтрація та очищення
│   └── fallback_handler.py    # Обробка fallback відповідей
├── chat/                # Чат інтерфейси (майбутнє розширення)
└── testing/             # Тестування (майбутнє розширення)
```

## Використання

### Основні команди

```bash
# Інтерактивний чат
python main.py

# Інтерактивний чат з debug інформацією
python main.py --debug

# Тестування системи з заданими промптами
python main.py --test

# Запуск unit тестів
python main.py --unit-tests

# Показати довідку
python main.py --help
```

### Приклад використання модулів

```python
from utils.data_loader import get_fallbacks_data
from generation.response_generator import ResponseGenerator
from generation.fallback_handler import FallbackHandler
from core.validation import validate_response_quality, validate_prompt

# Завантаження даних
fallbacks = get_fallbacks_data()

# Ініціалізація генератора
response_gen = ResponseGenerator()
fallback_handler = FallbackHandler()

# Генерація відповіді
response = response_gen.get_smart_fallback("Hello")
print(response)  # "Hello! Nice to meet you."

# Валідація відповіді
is_valid = validate_response_quality(response, source='fallback')
print(f"Response is valid: {is_valid}")

# Валідація промпта
prompt_result = validate_prompt("What's your name?")
print(f"Prompt validation: {prompt_result}")
```

## Конфігурація

Всі параметри зберігаються в `config.py`:

- `TRITON_MODEL_NAME` - назва моделі в Triton
- `TRITON_URL` - URL Triton сервера
- `TARGET_LENGTH` - цільова довжина відповіді
- `GENERATION_TEMPERATURE` - температура генерації
- `MODEL_NAME` - назва HuggingFace моделі

## Fallback дані

Файл `fallbacks.json` містить:

- `fallback_responses` - прості fallback відповіді
- `smart_fallbacks` - розумні fallback з варіантами
- `default_responses` - дефолтні відповіді
- `test_prompts` - тестові промпти
- `bad_patterns` - погані патерни для фільтрації
- `problematic_tokens` - проблематичні токени

## Особливості

### Модульна архітектура
- Чітке розділення відповідальності
- Легке тестування окремих компонентів
- Простота розширення функціональності

### Розумні fallback
- Визначення наміру користувача
- Контекстуальні відповіді
- Адаптивні відповіді з урахуванням історії

### Якісна фільтрація та валідація

**Логіка вибору рівня фільтрації базується на джерелі відповіді:**

1. **Визначення джерела відповіді (`source`):**
   - `'triton'` - відповідь згенерована Triton сервером
   - `'smart_fallback'` - розумна fallback відповідь 
   - `'simple_fallback'` - проста fallback відповідь
   - `'fallback_only'` - fallback коли Triton недоступний
   - `'emergency'` - аварійна fallback при помилках

2. **Вибір рівня валідації:**
   ```python
   if source in ['smart_fallback', 'simple_fallback', 'fallback_only']:
       # М'яка валідація для fallback відповідей
       is_valid = validate_response_quality(text, source='fallback')
   else:
       # Строга валідація для Triton відповідей  
       is_valid = validate_response_quality(text, source='triton')
   ```

3. **М'яка валідація (для fallback):**
   - ✅ Дозволяє `!` та `?` символи
   - ✅ Перевіряє тільки критичні погані патерни (`I I I`, `YOU YOU`)
   - ✅ Дозволяє до 7 повторних символів підряд
   - ❌ Блокує тільки екстремальні випадки

4. **Строга валідація (для Triton):**
   - ❌ Блокує всі проблематичні токени включно з `!`, `?`
   - ❌ Перевіряє всі погані патерни та якісні перевірки
   - ❌ Блокує 5+ повторних символів підряд
   - ❌ Контролює співвідношення великих букв (>70%)

**Автоматичний fallback:** Якщо Triton відповідь не пройшла строгу валідацію, система автоматично переключається на fallback з м'якою валідацією.

**Порівняння рівнів фільтрації:**

| Параметр | М'яка валідація (Fallback) | Строга валідація (Triton) |
|----------|---------------------------|---------------------------|
| Символи `!`, `?` | ✅ Дозволено | ❌ Заборонено |
| Погані патерни | Тільки критичні | Всі патерни |
| Повторні символи | До 7 підряд | До 5 підряд |
| Великі букви | Не перевіряється | Максимум 70% |
| Проблематичні токени | Ігноруються `!`, `?` | Всі заборонені |
| Мінімальна довжина | 1+ символів | 3+ символів |

### Обробка помилок
- Graceful fallback при недоступності Triton
- Аварійні відповіді при критичних помилках
- Інформативні повідомлення про стан системи

## Розробка

### Додавання нових модулів

1. Створіть новий файл в відповідній директорії
2. Додайте імпорт в `__init__.py`
3. Оновіть `main.py` при необхідності

### Тестування

Система підтримує кілька рівнів тестування:

- Unit тести для окремих модулів (`--unit-tests`)
- Системні тести з промптами (`--test`)
- Debug режим для детального логування (`--debug`)

### Налагодження

Використовуйте `--debug` для додаткової інформації:

```bash
python main.py --debug
```

Це покаже джерело відповіді та інформацію про якість.

## Залежності

- `numpy` - для математичних операцій
- `transformers` - для HuggingFace моделей
- `tritonclient` - для підключення до Triton Server
- Стандартна бібліотека Python

## Історія змін

### v2.0 (Модульна версія)
- Повна модуляризація коду
- Централізація конфігурації в JSON
- Покращені fallback механізми
- Розумна фільтрація відповідей
- Додано unit та системні тести

### v1.0 (Монолітна версія)
- Базова функціональність DialoGPT + Triton
- Прості fallback відповіді
- Початкова реалізація 


на майбутне  поставити градієн дря рівня фільтрації ти    від 0-10 
типу 0 мяка 10 хард 