import ollama

# Send a simple query to your local Ollama model
response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": "Hello Ollama, are you working with Python?"}
])

print("Ollama Response:")
print(response['message']['content'])
