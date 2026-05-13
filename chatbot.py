import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer user questions clearly and concisely. "
    "If a question is outside your knowledge or inappropriate, politely say so."
)


def chat():
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("Azure OpenAI Chatbot (GPT-4o)")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    print("\nAssistant: Hello! I'm your AI assistant powered by Azure OpenAI GPT-4o. How can I help you today?\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nAssistant: Goodbye! Have a great day!")
            break

        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=conversation_history,
                max_tokens=800,
                temperature=0.7,
            )

            assistant_message = response.choices[0].message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})

            print(f"\nAssistant: {assistant_message}\n")

        except Exception as e:
            print(f"\nError communicating with Azure OpenAI: {e}\n")
            conversation_history.pop()


if __name__ == "__main__":
    chat()
