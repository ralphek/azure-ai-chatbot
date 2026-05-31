"""
Document analysis using Azure AI Document Intelligence.

Supports: PDF, DOCX, XLSX, PNG, JPG, TIFF, BMP
Returns extracted text + tables as a plain string for GPT-4o to analyze.
"""
import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv(override=True)

ENDPOINT = os.getenv("DOC_INTELLIGENCE_ENDPOINT")
KEY = os.getenv("DOC_INTELLIGENCE_KEY")


def _client():
    return DocumentIntelligenceClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def analyze_file(file_path: str) -> str:
    """
    Extract text and tables from a local file.
    Returns a plain-text string ready to pass to GPT-4o.
    """
    if not os.path.exists(file_path):
        return f"File not found: {file_path}"

    client = _client()

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",  # handles text + tables in any doc type
        body=AnalyzeDocumentRequest(bytes_source=file_bytes),
    )
    result = poller.result()

    sections = []

    # Extract all text paragraphs
    if result.paragraphs:
        for para in result.paragraphs:
            text = para.content.strip()
            if text:
                sections.append(text)

    # Extract tables as markdown-style grids
    if result.tables:
        for i, table in enumerate(result.tables, 1):
            rows: dict[int, dict[int, str]] = {}
            for cell in table.cells:
                rows.setdefault(cell.row_index, {})[cell.column_index] = cell.content.strip()
            table_lines = [f"\n[Table {i}]"]
            for row_idx in sorted(rows):
                row_cells = [rows[row_idx].get(c, "") for c in range(table.column_count)]
                table_lines.append(" | ".join(row_cells))
            sections.append("\n".join(table_lines))

    return "\n\n".join(sections) if sections else "No readable content found in the document."


def analyze_url(url: str) -> str:
    """Extract text and tables from a publicly accessible URL."""
    client = _client()

    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(url_source=url),
    )
    result = poller.result()

    sections = []
    if result.paragraphs:
        for para in result.paragraphs:
            text = para.content.strip()
            if text:
                sections.append(text)
    if result.tables:
        for i, table in enumerate(result.tables, 1):
            rows: dict[int, dict[int, str]] = {}
            for cell in table.cells:
                rows.setdefault(cell.row_index, {})[cell.column_index] = cell.content.strip()
            table_lines = [f"\n[Table {i}]"]
            for row_idx in sorted(rows):
                row_cells = [rows[row_idx].get(c, "") for c in range(table.column_count)]
                table_lines.append(" | ".join(row_cells))
            sections.append("\n".join(table_lines))

    return "\n\n".join(sections) if sections else "No readable content found."
