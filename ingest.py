"""
Run once to embed finance documents and upload them to Azure AI Search.
Usage: python ingest.py
"""
import os
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from rag import create_index, get_embedding, _openai_client, SEARCH_ENDPOINT, SEARCH_KEY, INDEX_NAME
from dotenv import load_dotenv

load_dotenv()

DOCUMENTS = [
    {
        "id": "doc-pe-ratio",
        "source": "finance-glossary",
        "content": "P/E Ratio (Price-to-Earnings): A valuation metric calculated by dividing the stock price by earnings per share (EPS). A high P/E may indicate the stock is overvalued or that investors expect high growth. The S&P 500 average P/E ratio is typically between 15-25. Value investors prefer low P/E stocks (below 15), while growth stocks often trade at P/E ratios of 30 or higher.",
    },
    {
        "id": "doc-dca",
        "source": "finance-glossary",
        "content": "Dollar-Cost Averaging (DCA): An investment strategy where you invest a fixed amount at regular intervals regardless of the asset price. This reduces the impact of volatility. Example: investing $500 monthly into an S&P 500 ETF regardless of market conditions. Over time, you buy more shares when prices are low and fewer when prices are high, resulting in a lower average cost per share.",
    },
    {
        "id": "doc-emergency-fund",
        "source": "personal-finance",
        "content": "Emergency Fund: A financial safety net of 3-6 months of living expenses kept in a liquid, low-risk account such as a high-yield savings account. This fund protects against unexpected expenses like job loss, medical bills, or car repairs without going into debt. Most financial advisors recommend building this before investing.",
    },
    {
        "id": "doc-bull-bear",
        "source": "market-concepts",
        "content": "Bull Market vs Bear Market: A bull market is a period of rising stock prices (20%+ gain), typically associated with strong economic growth and investor confidence. A bear market is a sustained decline of 20% or more from recent highs, often associated with economic recession. The average bull market lasts about 4.5 years; the average bear market lasts about 9.6 months.",
    },
    {
        "id": "doc-index-funds",
        "source": "market-concepts",
        "content": "Index Funds and ETFs: Index funds track a market index like the S&P 500 (500 largest US companies). ETFs (Exchange-Traded Funds) are similar but trade like stocks throughout the day. Both offer diversification at low cost. The S&P 500 has historically returned approximately 10% annually before inflation. Warren Buffett recommends low-cost S&P 500 index funds for most investors.",
    },
    {
        "id": "doc-compound-interest",
        "source": "personal-finance",
        "content": "Compound Interest: The process where interest is earned on both the initial principal and previously accumulated interest. Example: $10,000 invested at 7% annually becomes $76,123 after 30 years. Starting early is critical — investing $200/month from age 25 to 65 at 7% yields approximately $525,000, while starting at 35 yields only $243,000.",
    },
    {
        "id": "doc-diversification",
        "source": "market-concepts",
        "content": "Diversification: Spreading investments across different asset classes, sectors, and geographies to reduce risk. A diversified portfolio might include US stocks, international stocks, bonds, and real estate. Modern Portfolio Theory shows diversification can reduce portfolio volatility without sacrificing returns.",
    },
    {
        "id": "doc-hysa",
        "source": "banking",
        "content": "High-Yield Savings Account (HYSA): A savings account offering significantly higher interest rates than traditional savings accounts. Rates in 2024 ranged from 4-5% APY compared to the national average of 0.46%. Typically offered by online banks. FDIC insured up to $250,000. Ideal for emergency funds and short-term savings goals.",
    },
    {
        "id": "doc-bonds",
        "source": "investing",
        "content": "Bonds: Debt instruments where an investor loans money to a government or corporation in exchange for periodic interest payments and return of principal at maturity. Government bonds (Treasuries) are considered low-risk. Corporate bonds offer higher yields but carry more risk. Bond prices move inversely to interest rates — when rates rise, bond prices fall.",
    },
    {
        "id": "doc-401k",
        "source": "retirement",
        "content": "401(k) Retirement Plan: Employer-sponsored retirement savings account allowing pre-tax contributions up to $23,000/year (2024 limit). Many employers match contributions up to a percentage — always contribute enough to get the full match (it's free money). Funds grow tax-deferred until withdrawal at retirement (age 59.5+). Early withdrawal incurs a 10% penalty plus income tax.",
    },
]


def ingest():
    print("=" * 50)
    print("FinanceBot RAG Ingestion")
    print("=" * 50)

    print("\n[1/3] Creating Azure AI Search index...")
    create_index()

    print("\n[2/3] Generating embeddings...")
    client = _openai_client()
    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY),
    )

    documents = []
    for doc in DOCUMENTS:
        print(f"  Embedding: {doc['id']}")
        vector = get_embedding(doc["content"], client)
        documents.append({
            "id": doc["id"],
            "content": doc["content"],
            "source": doc["source"],
            "content_vector": vector,
        })

    print(f"\n[3/3] Uploading {len(documents)} documents to Azure AI Search...")
    results = search_client.upload_documents(documents=documents)
    succeeded = sum(1 for r in results if r.succeeded)
    print(f"\nDone: {succeeded}/{len(documents)} documents indexed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    ingest()
