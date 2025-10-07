from interface.generator import generate_response

def start_chat():
    print("🤖 DialoGPT Chatbot (128 токенів контексту)")
    print("Введіть 'quit' для виходу")
    while True:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("👋 До побачення!")
            break
        if not user_input:
            continue
        response = generate_response(user_input)
        print(f"🤖 Bot: {response}")
