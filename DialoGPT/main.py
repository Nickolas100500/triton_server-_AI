from chat.interactiv import start_chat
from chat.test import run_tests
from pprint import pprint as pp

if __name__ == "__main__":
    try:
        pp("🎯 DialoGPT")
        pp("1. Чат")
        pp("2. Тест")
        choice = input("Ваш вибір: ").strip()

        if choice == "1":
            start_chat()
        elif choice == "2":
            run_tests()
        else:
            pp("❌ Невірний вибір")
            
    except KeyboardInterrupt:
        pp("\n\n👋 Програму закрито користувачем (Ctrl+C)")
    except Exception as e:
        pp(f"\n❌ Помилка: {e}")
    finally:
        pp("🔚 Завершення програми")
