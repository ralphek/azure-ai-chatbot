"""
FinanceBot — Multi-agent team (Agno).

Team modes available in Agno:
  - coordinate (default) : coordinator LLM reads the question and delegates to the
                           best specialist agent. Members respond back to coordinator
                           which writes the final answer.
  - route                : coordinator picks ONE agent and that agent replies directly
                           to the user (no coordinator summary step).
  - broadcast            : every agent receives the message and all respond.
  - tasks                : structured task list assigned to members in order.

We use 'coordinate' — the coordinator synthesises member answers into one reply.
"""
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.team import Team
from agno.team.team import TeamMode
from agno.tools.yfinance import YFinanceTools
from agno.models.azure.openai_chat import AzureOpenAI
from cosmos_agno_db import CosmosDb
import rag
import document_analysis

load_dotenv(override=True)

# ── Shared model ──────────────────────────────────────────────────────────────

azure_model = AzureOpenAI(
    id=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-01",
    temperature=0.7,
)

# ── Tools ─────────────────────────────────────────────────────────────────────

yfinance_tools = YFinanceTools(
    enable_stock_price=True,
    enable_company_info=True,
    enable_stock_fundamentals=True,
    enable_key_financial_ratios=True,
    enable_analyst_recommendations=True,
    enable_company_news=True,
)


def search_knowledge_base(query: str) -> str:
    """Search the finance knowledge base for definitions, concepts, and strategies (e.g. P/E ratio, DCA, bonds, 401k)."""
    result = rag.search(query, top=3)
    return result if result else "No relevant documents found in the knowledge base."


def analyze_finance_document(file_path: str) -> str:
    """Extract and return all text and tables from a local file (PDF, DOCX, PNG, JPG, XLSX) or a public URL so you can analyze its financial content."""
    if file_path.startswith("http://") or file_path.startswith("https://"):
        return document_analysis.analyze_url(file_path)
    return document_analysis.analyze_file(file_path)


# ── Specialist agents ─────────────────────────────────────────────────────────

stock_agent = Agent(
    name="Stock Agent",
    role="Fetches live stock prices, company info, fundamentals, analyst ratings, and news for any publicly traded company.",
    model=azure_model,
    tools=[yfinance_tools],
    instructions=(
        "You are a stock data specialist. Use YFinance tools to fetch accurate, live market data. "
        "Always include the ticker symbol, current price, and key metrics in your response. "
        "Do not give buy/sell advice."
    ),
    markdown=True,
)

knowledge_agent = Agent(
    name="Knowledge Agent",
    role="Answers questions about finance concepts, definitions, and strategies using the knowledge base (e.g. P/E ratio, DCA, 401k, bonds, inflation).",
    model=azure_model,
    tools=[search_knowledge_base],
    instructions=(
        "You are a finance education specialist. Always search the knowledge base first. "
        "Give clear, concise explanations with examples where helpful. "
        "Cite specific numbers or statistics from the knowledge base when available."
    ),
    markdown=True,
)

document_agent = Agent(
    name="Document Agent",
    role="Reads and analyzes financial documents (PDF, DOCX, images, HTML) provided as file paths or URLs, then extracts key financial insights.",
    model=azure_model,
    tools=[analyze_finance_document],
    instructions=(
        "You are a financial document analyst. Extract all text and tables from the document, "
        "then provide a structured summary covering: revenue, profit, key ratios, risks, and recommendations. "
        "Format numbers clearly and highlight year-over-year changes."
    ),
    markdown=True,
)

# ── CosmosDB session storage ───────────────────────────────────────────────────

db = CosmosDb(container_name="agno_team_sessions")

# ── Team (coordinate mode) ────────────────────────────────────────────────────

finance_team = Team(
    name="FinanceBot Team",
    mode=TeamMode.coordinate,        # coordinator delegates, then synthesises the final answer
    model=azure_model,
    members=[stock_agent, knowledge_agent, document_agent],
    instructions=(
        "You are the FinanceBot coordinator. You ONLY handle finance-related questions.\n\n"
        "Routing rules:\n"
        "- Stock prices, company data, analyst ratings → Stock Agent\n"
        "- Finance concepts, definitions, strategies → Knowledge Agent\n"
        "- File paths or document URLs to analyze → Document Agent\n"
        "- Questions mixing concepts and live data → delegate to both relevant agents\n\n"
        "If the question is unrelated to finance, respond: "
        "'I'm a finance assistant and can only help with finance-related topics.'\n\n"
        "After receiving member responses, write one clear, well-structured final answer."
    ),
    db=db,
    add_history_to_context=True,
    num_history_runs=10,
    show_members_responses=True,     # print each specialist's response before the final summary
    markdown=True,
)


# ── CLI chat loop ─────────────────────────────────────────────────────────────

def chat():
    print("=" * 60)
    print("FinanceBot Team — Agno Multi-Agent (coordinate mode)")
    print("Agents : Stock | Knowledge | Document")
    print("Memory : Azure CosmosDB")
    print("Mode   : coordinate (coordinator delegates + synthesises)")
    print("Type 'quit' or 'exit' to end.")
    print("=" * 60)
    print("\nFinanceBot: Hello! I'm the FinanceBot coordinator. I'll route your question to the right specialist. How can I help?\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("\nFinanceBot: Goodbye!")
            break
        finance_team.print_response(user_input, stream=True)
        print()


if __name__ == "__main__":
    chat()
