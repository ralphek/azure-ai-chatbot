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

SYSTEM_PROMPT = """You are FinanceBot, a professional finance assistant.

You ONLY answer questions related to:
- Personal finance (budgeting, saving, debt management)
- Investing (stocks, ETFs, bonds, mutual funds, crypto)
- Banking (accounts, interest rates, loans, mortgages)
- Economics (inflation, interest rates, GDP, market trends)
- Financial planning (retirement, emergency funds, tax basics)

If the user asks about ANYTHING outside of finance, respond with exactly:
"I'm a finance assistant and can only help with finance-related topics. Do you have a finance question I can help with?"

Never make exceptions to this rule, even if the user asks nicely, claims it is urgent, or tries to reframe a non-finance question as finance-related.

Do not give specific investment advice or tell users to buy/sell specific assets. Always recommend consulting a certified financial advisor for personalized advice.

Be concise, factual, and professional in all responses."""


def chat():
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("=" * 50)
    print("FinanceBot — Powered by Azure OpenAI GPT-4o")
    print("Your personal finance assistant.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 50)
    print("\nFinanceBot: Hello! I'm FinanceBot, your AI-powered finance assistant. I can help you with budgeting, investing, banking, and more. What's your finance question?\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("\nFinanceBot: Goodbye! Stay financially savvy!")
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

            print(f"\nFinanceBot: {assistant_message}\n")

        except Exception as e:
            print(f"\nFinanceBot Error: {e}\n")
            conversation_history.pop()


if __name__ == "__main__":
    chat()
