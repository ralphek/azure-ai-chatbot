import os
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

load_dotenv()


class CosmosMemory:
    """Persistent conversation memory backed by Azure Cosmos DB."""

    def __init__(self):
        self.client = CosmosClient(
            os.getenv("COSMOS_ENDPOINT"),
            os.getenv("COSMOS_KEY"),
        )
        db = self.client.create_database_if_not_exists(
            id=os.getenv("COSMOS_DATABASE", "financebot")
        )
        self.container = db.create_container_if_not_exists(
            id=os.getenv("COSMOS_CONTAINER", "memory"),
            partition_key=PartitionKey(path="/userId"),
        )

    def load(self, user_id: str = "default") -> list:
        try:
            item = self.container.read_item(item=user_id, partition_key=user_id)
            return item.get("messages", [])
        except exceptions.CosmosResourceNotFoundError:
            return []

    def save(self, history: list, user_id: str = "default") -> None:
        messages = [
            m for m in history
            if isinstance(m, dict) and m.get("role") != "system"
        ]
        self.container.upsert_item({
            "id": user_id,
            "userId": user_id,
            "messages": messages,
        })

    def clear(self, user_id: str = "default") -> None:
        try:
            self.container.delete_item(item=user_id, partition_key=user_id)
        except exceptions.CosmosResourceNotFoundError:
            pass
