# Architecture: Basic Azure OpenAI Chatbot

## Overview

This document describes the architecture of the basic Azure OpenAI chatbot built in Lab 1.

## Architecture Diagram

```
┌─────────────┐        ┌──────────────────────┐        ┌────────────────────────────┐
│    User     │ ──────▶│  Python Application   │ ──────▶│     Azure OpenAI Service   │
│  (Terminal) │        │    (chatbot.py)        │        │                            │
│             │◀────── │                        │◀────── │  Resource Group            │
└─────────────┘        │  • Load .env config    │        │  └── OpenAI Resource       │
                       │  • Maintain history    │        │       └── GPT-4o Deployment│
                       │  • Handle errors       │        │                            │
                       └──────────────────────┘        └────────────────────────────┘
```

## Data Flow

1. **User Input** — The user types a message in the terminal.
2. **Application Layer** (`chatbot.py`) — Appends the message to the running conversation history and calls the Azure OpenAI API.
3. **Azure OpenAI Service** — Receives the full conversation history, processes it through the GPT-4o model, and returns a completion.
4. **Response** — The application appends the assistant reply to history and prints it to the user.

## Azure Resources

| Resource | Purpose |
|---|---|
| Resource Group | Logical container for all related Azure resources |
| Azure OpenAI Resource | Hosts the OpenAI service endpoint and API keys |
| GPT-4o Deployment | The deployed model that handles chat completions |

## Design Decisions

- **Stateful conversation history** — The full message list is kept in memory and sent on every API call, giving the model context of the whole session without requiring a database.
- **Environment variables for secrets** — API keys and endpoint URLs are loaded from a `.env` file (excluded from source control via `.gitignore`) so credentials are never committed.
- **System prompt** — A concise system prompt establishes assistant behaviour and graceful handling of out-of-scope questions.
- **Single-file simplicity** — For Lab 1, all logic lives in `chatbot.py` to keep the structure easy to follow before complexity is added in later labs.

## Component Interactions

```
chatbot.py
 ├── load_dotenv()           → reads AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT
 ├── AzureOpenAI(client)     → authenticated connection to Azure OpenAI
 └── chat()
      ├── system message      → sets assistant persona
      ├── input loop          → collects user messages
      ├── client.chat.completions.create() → sends history to GPT-4o, receives reply
      └── error handling      → catches API exceptions and continues the loop
```

## Security Notes

- `.env` is listed in `.gitignore` — never commit it.
- Use Azure Key Vault or environment secrets in production instead of a local `.env` file.
