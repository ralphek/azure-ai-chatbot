import os
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
)

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

CONTENT_SAFETY_ENDPOINT = os.getenv("CONTENT_SAFETY_ENDPOINT")
CONTENT_SAFETY_KEY = os.getenv("CONTENT_SAFETY_KEY")

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


def is_content_safe(text):
    """
    Layer 2: Sends text to Azure AI Content Safety.
    Returns (is_safe: bool, reason: str).
    If Content Safety is not configured, skips and allows the message through.
    """
    if not CONTENT_SAFETY_ENDPOINT or not CONTENT_SAFETY_KEY:
        return True, "Content Safety not configured — skipping."

    url = (
        f"{CONTENT_SAFETY_ENDPOINT.rstrip('/')}"
        "/contentsafety/text:analyze?api-version=2023-10-01"
    )
    headers = {
        "Ocp-Apim-Subscription-Key": CONTENT_SAFETY_KEY,
        "Content-Type": "application/json",
    }
    body = {
        "text": text,
        "categories": ["Hate", "Violence", "Sexual", "SelfHarm"],
        "blocklistNames": [],
    }

    try:
        response = requests.post(url, headers=headers, json=body, timeout=5)
        response.raise_for_status()
        result = response.json()

        for category in result.get("categoriesAnalysis", []):
            if category["severity"] >= 4:
                return False, f"Blocked by Content Safety: {category['category']} (severity {category['severity']})"

        return True, "OK"

    except requests.exceptions.RequestException as e:
        print(f"[Content Safety] Warning: could not reach service ({e}). Allowing message through.")
        return True, "Content Safety unreachable — skipping."


def chat():
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

    content_safety_active = bool(CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_KEY)

    print("=" * 50)
    print("FinanceBot — Powered by Azure OpenAI GPT-4o")
    print("Your personal finance assistant.")
    print(f"Content Safety (Layer 2): {'ACTIVE' if content_safety_active else 'NOT CONFIGURED'}")
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

        # --- Layer 2: Azure AI Content Safety ---
        safe, reason = is_content_safe(user_input)
        if not safe:
            print(f"\nFinanceBot: I'm not able to respond to that message. Please keep the conversation respectful and on topic.\n")
            continue

        # --- Layer 1: GPT-4o with finance-only system prompt ---
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
