import ollama

print("🤖 LocalBot (Ollama) is ready! Type 'exit' to quit.\n")

conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Add user message to conversation
    conversation.append({"role": "user", "content": user_input})

    # Send to Ollama
    response = ollama.chat(model="llama3", messages=conversation)

    # Get reply text
    bot_reply = response['message']['content']
    print("Bot:", bot_reply)

    # Add bot reply to conversation
    conversation.append({"role": "assistant", "content": bot_reply})
