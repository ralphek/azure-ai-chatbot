"""
Azure CosmosDB backend for Agno's BaseDb interface.

Replaces SqliteDb so agent sessions (full conversation history + runs)
are stored in CosmosDB instead of a local SQLite file.

Only session read/write is implemented — the other abstract methods
(memories, knowledge, traces, evals, etc.) are stubbed as safe no-ops
since FinanceBot does not use those features.
"""
import json
import time
import os
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

from agno.db.base import BaseDb
from agno.db.utils import CustomJSONEncoder
from agno.agent.agent import AgentSession

load_dotenv(override=True)


class CosmosDb(BaseDb):
    """Agno BaseDb backed by Azure CosmosDB NoSQL API."""

    def __init__(
        self,
        container_name: str = "agno_sessions",
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        database_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        endpoint = endpoint or os.getenv("COSMOS_ENDPOINT")
        key = key or os.getenv("COSMOS_KEY")
        database_name = database_name or os.getenv("COSMOS_DATABASE", "financebot")

        client = CosmosClient(endpoint, key)
        db = client.create_database_if_not_exists(id=database_name)
        self._container = db.create_container_if_not_exists(
            id=container_name,
            partition_key=PartitionKey(path="/session_id"),
        )
        print(f"[CosmosDb] Connected to {database_name}/{container_name}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _to_doc(self, session: AgentSession) -> Dict[str, Any]:
        """Serialize AgentSession → CosmosDB document."""
        raw = session.to_dict()
        # Round-trip through JSON to flatten any non-serializable types
        doc = json.loads(json.dumps(raw, cls=CustomJSONEncoder))
        doc["id"] = doc.get("session_id") or str(uuid4())
        return doc

    def _from_doc(self, doc: Dict[str, Any]) -> Optional[AgentSession]:
        """Deserialize CosmosDB document → AgentSession."""
        for field in ("_rid", "_self", "_etag", "_attachments", "_ts"):
            doc.pop(field, None)
        return AgentSession.from_dict(doc)

    # ── Required by BaseDb ────────────────────────────────────────────────────

    def table_exists(self, table_name: str) -> bool:
        return True  # container created in __init__

    # ── Session CRUD (actually used by Agent) ─────────────────────────────────

    def upsert_session(self, session, deserialize=True):
        try:
            doc = self._to_doc(session)
            doc["updated_at"] = int(time.time())
            result = dict(self._container.upsert_item(doc))
            return self._from_doc(result) if deserialize else result
        except Exception as e:
            print(f"[CosmosDb] upsert_session error: {e}")
            raise

    def upsert_sessions(self, sessions, deserialize=True):
        return [self.upsert_session(s, deserialize=deserialize) for s in sessions]

    def get_session(self, session_id, session_type=None, user_id=None, deserialize=True):
        try:
            doc = dict(self._container.read_item(item=session_id, partition_key=session_id))
            return self._from_doc(doc) if deserialize else doc
        except exceptions.CosmosResourceNotFoundError:
            return None
        except Exception as e:
            print(f"[CosmosDb] get_session error: {e}")
            return None

    def get_sessions(self, session_type=None, user_id=None, agent_id=None,
                     workflow_id=None, team_id=None, deserialize=True):
        try:
            query = "SELECT * FROM c"
            conditions, params = [], []
            if session_type is not None:
                val = session_type.value if hasattr(session_type, "value") else session_type
                conditions.append("c.session_type = @st")
                params.append({"name": "@st", "value": val})
            if user_id is not None:
                conditions.append("c.user_id = @uid")
                params.append({"name": "@uid", "value": user_id})
            if agent_id is not None:
                conditions.append("c.agent_id = @aid")
                params.append({"name": "@aid", "value": agent_id})
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            items = list(self._container.query_items(
                query=query,
                parameters=params or None,
                enable_cross_partition_query=True,
            ))
            if not deserialize:
                return items
            return [s for s in (self._from_doc(dict(d)) for d in items) if s]
        except Exception as e:
            print(f"[CosmosDb] get_sessions error: {e}")
            return []

    def delete_session(self, session_id):
        try:
            self._container.delete_item(item=session_id, partition_key=session_id)
        except exceptions.CosmosResourceNotFoundError:
            pass
        except Exception as e:
            print(f"[CosmosDb] delete_session error: {e}")

    def delete_sessions(self, session_type=None, user_id=None, agent_id=None):
        for doc in self.get_sessions(session_type=session_type, user_id=user_id,
                                      agent_id=agent_id, deserialize=False):
            sid = doc.get("session_id") or doc.get("id")
            if sid:
                self.delete_session(sid)

    def rename_session(self, session_id, new_name, user_id=None):
        session = self.get_session(session_id, user_id=user_id)
        if session and session.session_data:
            session.session_data["session_name"] = new_name
            self.upsert_session(session)

    # ── Stub all other abstract methods ───────────────────────────────────────

    def calculate_metrics(self, *a, **kw): return None
    def clear_cultural_knowledge(self, *a, **kw): pass
    def clear_memories(self, *a, **kw): pass
    def create_eval_run(self, *a, **kw): return None
    def create_span(self, *a, **kw): return None
    def create_spans(self, *a, **kw): return None
    def delete_cultural_knowledge(self, *a, **kw): pass
    def delete_eval_runs(self, *a, **kw): pass
    def delete_knowledge_content(self, *a, **kw): pass
    def delete_learning(self, *a, **kw): pass
    def delete_user_memories(self, *a, **kw): pass
    def delete_user_memory(self, *a, **kw): pass
    def get_all_cultural_knowledge(self, *a, **kw): return []
    def get_all_memory_topics(self, *a, **kw): return []
    def get_cultural_knowledge(self, *a, **kw): return None
    def get_eval_run(self, *a, **kw): return None
    def get_eval_runs(self, *a, **kw): return []
    def get_knowledge_content(self, *a, **kw): return None
    def get_knowledge_contents(self, *a, **kw): return []
    def get_latest_schema_version(self, *a, **kw): return None
    def get_learning(self, *a, **kw): return None
    def get_learnings(self, *a, **kw): return []
    def get_metrics(self, *a, **kw): return None
    def get_span(self, *a, **kw): return None
    def get_spans(self, *a, **kw): return []
    def get_trace(self, *a, **kw): return None
    def get_trace_stats(self, *a, **kw): return None
    def get_traces(self, *a, **kw): return []
    def get_user_memories(self, *a, **kw): return []
    def get_user_memory(self, *a, **kw): return None
    def get_user_memory_stats(self, *a, **kw): return None
    def rename_eval_run(self, *a, **kw): pass
    def upsert_cultural_knowledge(self, *a, **kw): return None
    def upsert_knowledge_content(self, *a, **kw): return None
    def upsert_learning(self, *a, **kw): return None
    def upsert_memories(self, *a, **kw): return None
    def upsert_schema_version(self, *a, **kw): return None
    def upsert_trace(self, *a, **kw): return None
    def upsert_user_memory(self, *a, **kw): return None
