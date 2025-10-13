
from interface.generator import generate_response

def run_tests():
    """Тест роботи з 128 токенами"""
    test_prompts = [
        "Hello, how are you doing today?",
        "What's your favorite hobby?", 
        "Can you tell me about artificial intelligence?",
        "What do you think about the weather?"
    ]
    
    print("🧪 ТЕСТ З 128 ТОКЕНАМИ КОНТЕКСТУ")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts):
        try:
            print(f"\n{i+1}. 👤 User: {prompt}")
            print("🤖 Bot: ", end="", flush=True)
            
            response = generate_response(prompt, max_new_tokens=25)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n⏹️ Тест перервано")
            break
        except Exception as e:
            print(f"❌ Помилка: {e}")