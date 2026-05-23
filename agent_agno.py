"""
FinanceBot — Agno-based agent.
Replaces the manual tool loop, conversation history, and RAG injection in chatbot.py.
"""
import os
import json
import yfinance as yf
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure.openai_chat import AzureOpenAI
from agno.db.sqlite import SqliteDb
import rag

load_dotenv(override=True)

SYSTEM_PROMPT = """You are FinanceBot, a professional finance assistant.

You ONLY answer questions related to:
- Personal finance (budgeting, saving, debt management)
- Investing (stocks, ETFs, bonds, mutual funds, crypto)
- Banking (accounts, interest rates, loans, mortgages)
- Economics (inflation, interest rates, GDP, market trends)
- Financial planning (retirement, emergency funds, tax basics)

When the user asks about a stock price or company stock, always use the get_stock_price tool.
When the user asks about a finance concept (P/E ratio, DCA, bonds, etc.), use the search_knowledge_base tool first.

If the user asks about ANYTHING outside of finance, respond with exactly:
"I'm a finance assistant and can only help with finance-related topics. Do you have a finance question I can help with?"

Do not give specific investment advice or tell users to buy/sell specific assets.
Be concise, factual, and professional."""


def get_stock_price(ticker: str) -> str:
    """Get the current real-time stock price and basic info for a given ticker symbol (e.g. AAPL, MSFT, TSLA)."""
    try:
        stock = yf.Ticker(ticker.upper())
        info = stock.info
        result = {
            "ticker": ticker.upper(),
            "company": info.get("shortName", "Unknown"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "currency": info.get("currency", "USD"),
            "change_percent": round(info.get("regularMarketChangePercent", 0), 2),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": f"Could not fetch data for {ticker}: {str(e)}"})


def search_knowledge_base(query: str) -> str:
    """Search the finance knowledge base for relevant information about finance concepts, definitions, and strategies."""
    context = rag.search(query, top=3)
    return context if context else "No relevant documents found in the knowledge base."


db = SqliteDb(db_file="agno_sessions.db")

agent = Agent(
    model=AzureOpenAI(
        id=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-02-01",
        temperature=0.7,
    ),
    tools=[get_stock_price, search_knowledge_base],
    instructions=SYSTEM_PROMPT,
    db=db,
    add_history_to_context=True,
    num_history_runs=10,
    show_tool_calls=True,
    markdown=True,
)


def chat():
    print("=" * 50)
    print("FinanceBot — Powered by Agno + Azure OpenAI")
    print("Knowledge base: Azure AI Search (RAG)")
    print("Session storage: SQLite (agno_sessions.db)")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 50)

    greeting = "Hello! I'm FinanceBot — I can answer finance questions and look up live stock prices. What can I help you with?"
    print(f"\nFinanceBot: {greeting}\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nFinanceBot: Goodbye! Stay financially savvy!")
            break

        agent.print_response(user_input, stream=True)
        print()


if __name__ == "__main__":
    chat()
