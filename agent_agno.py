"""
FinanceBot — Agno-based agent.

What Agno replaces vs chatbot.py:
  - YFinanceTools     → replaces manual get_stock_price() function
  - SqliteDb          → replaces CosmosMemory / memory.json
  - Agent.run()       → replaces manual tool call loop + conversation_history
  - search_knowledge_base tool → wraps Azure AI Search RAG
"""
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure.openai_chat import AzureOpenAI
from agno.tools.yfinance import YFinanceTools
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

When the user asks about a stock, use the YFinance tools to fetch live data.
When the user asks about a finance concept, use search_knowledge_base first.

If the user asks about ANYTHING outside of finance, respond with exactly:
"I'm a finance assistant and can only help with finance-related topics. Do you have a finance question I can help with?"

Do not give specific investment advice or tell users to buy/sell specific assets.
Be concise, factual, and professional."""


def search_knowledge_base(query: str) -> str:
    """Search the finance knowledge base for definitions, concepts, and strategies (e.g. P/E ratio, DCA, bonds, 401k)."""
    context = rag.search(query, top=3)
    return context if context else "No relevant documents found in the knowledge base."


# ── Agno skillsets replacing manual Python code ──────────────────────────────

# YFinanceTools replaces the 25-line manual get_stock_price() function.
# It provides 6 built-in tools the agent can call as needed.
yfinance_tools = YFinanceTools(
    enable_stock_price=True,           # current price
    enable_company_info=True,          # name, sector, description
    enable_stock_fundamentals=True,    # P/E, EPS, market cap
    enable_key_financial_ratios=True,  # ROE, debt-to-equity, etc.
    enable_analyst_recommendations=True,  # buy/hold/sell ratings
    enable_company_news=True,          # latest headlines
)

# SqliteDb replaces CosmosMemory — stores full session history locally.
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
    tools=[yfinance_tools, search_knowledge_base],
    instructions=SYSTEM_PROMPT,
    db=db,
    add_history_to_context=True,
    num_history_runs=10,
    markdown=True,
)


def chat():
    print("=" * 55)
    print("FinanceBot — Powered by Agno + Azure OpenAI")
    print("Skillsets : YFinanceTools (stock, fundamentals, news)")
    print("Knowledge : Azure AI Search RAG")
    print("Memory    : SQLite session storage (Agno built-in)")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 55)
    print("\nFinanceBot: Hello! I'm FinanceBot. I can look up live stock prices, fundamentals, analyst ratings, and answer finance questions. How can I help?\n")

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
