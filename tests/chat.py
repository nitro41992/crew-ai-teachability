from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def chat():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]

    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'quit':
            print("Chat terminated by the user.")
            break

        messages.append({"role": "user", "content": user_input})

        stream = client.chat.completions.create(
            messages=messages,
            model="llama3-70b-8192",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        print("Assistant: ", end="")
        for chunk in stream:
            print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

        messages.append({"role": "assistant", "content": chunk.choices[0].delta.content})

if __name__ == "__main__":
    print("Welcome to the Groq Chat Interface!")
    print("You can start chatting with the assistant now.")
    print("Type 'quit' to exit the chat.")
    print()

    try:
        chat()
    except KeyboardInterrupt:
        print("\nChat terminated by the user.")