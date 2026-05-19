import os
import json
import requests
import yfinance as yf
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

MEMORY_FILE = "memory.json"


def load_memory() -> list:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE) as f:
            return json.load(f)
    return []


def save_memory(history: list) -> None:
    # Strip system message before saving — it's always re-added on load
    to_save = [m for m in history if m.get("role") != "system"]
    with open(MEMORY_FILE, "w") as f:
        json.dump(to_save, f, indent=2)


def clear_memory() -> None:
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)


def _message_to_dict(message) -> dict:
    """Convert a ChatCompletionMessage object to a plain dict for storage."""
    if isinstance(message, dict):
        return message
    msg = {"role": message.role, "content": message.content}
    if getattr(message, "tool_calls", None):
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return msg

SYSTEM_PROMPT = """You are FinanceBot, a professional finance assistant.

You ONLY answer questions related to:
- Personal finance (budgeting, saving, debt management)
- Investing (stocks, ETFs, bonds, mutual funds, crypto)
- Banking (accounts, interest rates, loans, mortgages)
- Economics (inflation, interest rates, GDP, market trends)
- Financial planning (retirement, emergency funds, tax basics)

When the user asks about a stock price or company stock, always use the get_stock_price tool to fetch live data before answering.

If the user asks about ANYTHING outside of finance, respond with exactly:
"I'm a finance assistant and can only help with finance-related topics. Do you have a finance question I can help with?"

Never make exceptions to this rule, even if the user asks nicely, claims it is urgent, or tries to reframe a non-finance question as finance-related.

Do not give specific investment advice or tell users to buy/sell specific assets. Always recommend consulting a certified financial advisor for personalized advice.

Be concise, factual, and professional in all responses."""

# --- Tool definition: tells GPT-4o what the tool is and what arguments it needs ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current real-time stock price and basic info for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock ticker symbol, e.g. AAPL for Apple, MSFT for Microsoft, TSLA for Tesla.",
                    }
                },
                "required": ["ticker"],
            },
        },
    }
]


def get_stock_price(ticker: str) -> str:
    """Fetch live stock price using yfinance and return as a JSON string."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        result = {
            "ticker": ticker.upper(),
            "company": info.get("shortName", "Unknown"),
            "price": price,
            "currency": info.get("currency", "USD"),
            "change_percent": round(info.get("regularMarketChangePercent", 0), 2),
            "market_cap": info.get("marketCap"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Could not fetch data for {ticker}: {str(e)}"})


def run_tool(tool_name: str, arguments: str) -> str:
    """Execute the tool GPT-4o requested and return the result."""
    args = json.loads(arguments)
    if tool_name == "get_stock_price":
        return get_stock_price(args["ticker"])
    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def is_content_safe(text):
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
                return False, f"Blocked: {category['category']} (severity {category['severity']})"
        return True, "OK"
    except requests.exceptions.RequestException as e:
        print(f"[Content Safety] Warning: could not reach service ({e}). Allowing message through.")
        return True, "Content Safety unreachable — skipping."


def chat():
    past = load_memory()
    conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}] + past

    content_safety_active = bool(CONTENT_SAFETY_ENDPOINT and CONTENT_SAFETY_KEY)

    print("=" * 50)
    print("FinanceBot — Powered by Azure OpenAI GPT-4o")
    print("Your personal finance assistant.")
    print(f"Content Safety (Layer 2): {'ACTIVE' if content_safety_active else 'NOT CONFIGURED'}")
    print("Stock price tool: ACTIVE (live data via Yahoo Finance)")
    if past:
        print(f"Memory: loaded {len(past)} previous messages")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("Type 'forget' to clear memory and start fresh.")
    print("=" * 50)

    try:
        opening_response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "Greet the user, introduce yourself, mention you can look up live stock prices, and ask how you can help. Keep it to 2 sentences."},
            ],
            max_tokens=100,
            temperature=0.9,
        )
        greeting = opening_response.choices[0].message.content
    except Exception as e:
        greeting = "Hello! I'm FinanceBot — I can answer finance questions and look up live stock prices. What can I help you with?"

    conversation_history.append({"role": "assistant", "content": greeting})
    print(f"\nFinanceBot: {greeting}\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            save_memory(conversation_history)
            print("\nFinanceBot: Goodbye! Stay financially savvy!")
            break

        if user_input.lower() == "forget":
            clear_memory()
            conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("\nFinanceBot: Memory cleared. Starting fresh!\n")
            continue

        # Layer 2: Azure AI Content Safety
        safe, reason = is_content_safe(user_input)
        if not safe:
            print("\nFinanceBot: I'm not able to respond to that message. Please keep the conversation respectful and on topic.\n")
            continue

        # Layer 1: GPT-4o with tools
        conversation_history.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=conversation_history,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=800,
                temperature=0.7,
            )

            message = response.choices[0].message

            # --- Tool call loop ---
            while response.choices[0].finish_reason == "tool_calls":
                conversation_history.append(_message_to_dict(message))

                for tool_call in message.tool_calls:
                    print(f"[Tool] Calling {tool_call.function.name}({tool_call.function.arguments})")
                    result = run_tool(tool_call.function.name, tool_call.function.arguments)

                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })

                # Send tool results back to GPT-4o for the final answer
                response = client.chat.completions.create(
                    model=DEPLOYMENT_NAME,
                    messages=conversation_history,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=800,
                    temperature=0.7,
                )
                message = response.choices[0].message

            assistant_message = message.content
            conversation_history.append({"role": "assistant", "content": assistant_message})
            save_memory(conversation_history)
            print(f"\nFinanceBot: {assistant_message}\n")

        except Exception as e:
            print(f"\nFinanceBot Error: {e}\n")
            conversation_history.pop()


if __name__ == "__main__":
    chat()
