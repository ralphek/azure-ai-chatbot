# Azure AI Chatbot

A basic conversational chatbot powered by Azure OpenAI GPT-4o, built as part of Lab 1.

## Features

- Multi-turn conversation with full history context
- Welcome message on startup
- Graceful error handling
- Clean exit with `quit` or `exit`

## Prerequisites

- Python 3.8+
- Azure OpenAI resource with a GPT-4o deployment

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/azure-ai-chatbot.git
cd azure-ai-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=https://YOUR_RESOURCE.openai.azure.com/
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

Find these values in the Azure Portal under your OpenAI resource, Keys and Endpoint.

### 4. Run the chatbot

```bash
python chatbot.py
```

## API Configuration

| Variable | Description |
|---|---|
| AZURE_OPENAI_ENDPOINT | Your Azure OpenAI resource endpoint URL |
| AZURE_OPENAI_API_KEY | API key from Azure Portal |
| AZURE_OPENAI_DEPLOYMENT | Name of your GPT-4o deployment |

## Architecture

See [architecture.md](architecture.md) for the full architecture diagram and design decisions.

## License

MIT
