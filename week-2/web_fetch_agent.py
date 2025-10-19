# - Autonomous tool choice:
#     • URL in message  → fetch → summarize → save (DB + Vector store)
#     • Research query  → RAG (semantic local_search) → web search fallback (DuckDuckGo)
# - SQLite persistence for summaries
# - ChromaDB vector store (OpenAI embeddings) for semantic retrieval (RAG)


from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

# Built-in web search tool (DuckDuckGo)
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool

# --- Chroma for vector storage / semantic retrieval ---
import chromadb
from chromadb.utils import embedding_functions



DB_PATH = "summaries.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS page_summaries (
            url TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

init_db()


RAG_PATH = "rag_store"

def init_vector_store() -> chromadb.Collection:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is required for embeddings. Export it before running."
        )

    client = chromadb.PersistentClient(path=RAG_PATH)
    collection = client.get_or_create_collection(
        name="summaries",
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name="text-embedding-3-small",
        ),
    )
    return collection

@dataclass
class Deps:
    db_path: str = DB_PATH
    collection: Optional[chromadb.Collection] = None


def extract_text_and_title(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    # remove noise
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.body or soup
    text = "\n".join(
        line.strip() for line in body.get_text("\n").splitlines() if line.strip()
    )
    return text, title

class FetchedPage(BaseModel):
    url: str
    title: str
    text: str

class SavedSummary(BaseModel):
    url: str
    title: str
    saved: bool

class LocalDoc(BaseModel):
    url: str
    title: str
    snippet: str

AGENT_INSTRUCTIONS = r"""
You are a PRINCIPAL-LEVEL web research agent. Autonomously choose and call the right tool(s) based on the user message.

TOOLS AVAILABLE
- fetch_web_page(url): Fetches a URL, returns title+text.
- save_summary(url, title, summary): Persists a concise summary (DB) and updates the vector store.
- local_search(query, limit): Semantic retrieval (RAG) over stored summaries using embeddings; returns the most relevant items.
- duckduckgo_search: Web search when local data is insufficient.

DECISION POLICY (ALWAYS FOLLOW)
1) If the message contains one or more HTTP/HTTPS URLs:
   - For each URL (up to 10, in order):
     a) fetch_web_page(url)
     b) Summarize in 5–10 bullets (factual; from the fetched text only)
     c) save_summary(url, title, summary)
   - Return a section per URL with:
     • “Saved” confirmation (only after save_summary succeeds)
     • Title
     • URL
     • The 5–10 bullet summary

2) If the message is a research-style question or asks for external facts
   (e.g., “what are/why/how many/where…”, populations, threats, trends, data):
   - First call local_search(query) for semantic retrieval.
     • If local results exist: synthesize the answer ONLY from those local results,
       and include a **Sources** list using the stored titles + URLs returned by local_search.
     • If local results are empty: call duckduckgo_search to gather web results,
       synthesize the answer from those, and include a **Sources** list with titles + URLs.
   - Produce 5–10 concise bullets plus **Sources**.

3) If the user asks to “index”, “ingest”, or “save” pages without explicit URLs, ask for URLs; otherwise proceed with (2).

RULES
- Do not rely on prior model knowledge for URL summaries—always fetch first.
- Summarize only from the data actually retrieved (fetched page; local RAG results; or web results).
- Never fabricate data or sources. If a fetch/search fails, say so for that item and continue with others when possible.
- Keep answers concise, neutral, and factual. Prefer bullet lists.

OUTPUT FORMAT
- For URL flows: “Saved” line + Title + URL + 5–10 bullets (per URL).
- For research flows (RAG/web): 5–10 bullets + “**Sources:**” list with titles and URLs.
"""


duckduckgo = duckduckgo_search_tool()

async def logged_duckduckgo_search(ctx: RunContext[Deps], query: str, **kwargs):
    print(f"\n🔎 [DuckDuckGo Search Triggered] → Query: {query}")
    return await duckduckgo.func(ctx, query=query, **kwargs)

# Create the agent with the wrapped tool
agent = Agent(
    "openai:gpt-4o-mini",
    system_prompt=AGENT_INSTRUCTIONS,
    tools=[logged_duckduckgo_search],
)



@agent.tool
async def fetch_web_page(ctx: RunContext[Deps], url: str) -> FetchedPage:
    """
    Fetch Web Page — download and extract readable text + title.
    """
    async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
        r = await client.get(url, headers={"User-Agent": "pydantic-ai-agent/1.0"})
        r.raise_for_status()
    print(f"\n [Fetching Web Page] → {url}")
    text, title = extract_text_and_title(r.text)
    if len(text) > 80_000:
        text = text[:80_000] + "\n...[truncated]"
    return FetchedPage(url=url, title=title or url, text=text)


@agent.tool
def save_summary(ctx: RunContext[Deps], url: str, title: str, summary: str) -> SavedSummary:
    """
    Save Summary — store/update the summary in SQLite and upsert into Chroma for RAG.
    """
    # 1) Persist in SQLite
    conn = sqlite3.connect(ctx.deps.db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO page_summaries (url, title, summary, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
          title=excluded.title,
          summary=excluded.summary,
          created_at=excluded.created_at
        """,
        (url, title, summary, datetime.utcnow().isoformat(timespec="seconds")),
    )
    print(f"\n [Saving Summary] → {url}")
    conn.commit()
    conn.close()

    # 2) Upsert into Chroma (vector store)
    if ctx.deps.collection is None:
        # Lazily initialize if not provided in deps
        ctx.deps.collection = init_vector_store()

    ctx.deps.collection.upsert(
        ids=[url],
        documents=[summary],
        metadatas=[{"title": title, "url": url}],
    )

    return SavedSummary(url=url, title=title, saved=True)


@agent.tool
def local_search(ctx: RunContext[Deps], query: str, limit: int = 5) -> List[LocalDoc]:
    """
    Semantic RAG search — retrieve relevant stored summaries via vector similarity.
    Returns a ranked list of LocalDoc (url, title, snippet).
    """
    # Ensure collection exists
    print(f"\n [Local Semantic Search Triggered] → Query: {query}")
    if ctx.deps.collection is None:
        ctx.deps.collection = init_vector_store()

    results = ctx.deps.collection.query(query_texts=[query], n_results=limit or 5)

    docs: List[LocalDoc] = []
    if not results or not results.get("documents"):
        return docs

    retrieved_docs = results["documents"][0] or []
    metadatas = results["metadatas"][0] or []

    for doc, meta in zip(retrieved_docs, metadatas):
        url = (meta or {}).get("url", "")
        title = (meta or {}).get("title", "") or url
        snippet = (doc or "")[:300]
        if len(doc or "") > 300:
            snippet += "..."
        docs.append(LocalDoc(url=url, title=title, snippet=snippet))
    return docs


EXAMPLE_SINGLE = "What is this page about? https://en.wikipedia.org/wiki/Capybara"

EXAMPLE_BATCH = """Index these pages:
Lesser capybara — https://en.wikipedia.org/wiki/Lesser_capybara
Hydrochoerus (genus) — https://en.wikipedia.org/wiki/Hydrochoerus
Neochoerus (extinct genus related to capybaras) — https://en.wikipedia.org/wiki/Neochoerus
Caviodon (extinct genus of rodents related to capybaras) — https://en.wikipedia.org/wiki/Caviodon
Neochoerus aesopi (extinct species close to capybaras) — https://en.wikipedia.org/wiki/Neochoerus_aesopi
"""

EXAMPLE_QUESTION = "What are threats to capybara populations?"

async def main():
    # Initialize deps with a ready vector store so first call is fast
    deps = Deps(collection=init_vector_store())

    res1 = await agent.run(EXAMPLE_SINGLE, deps=deps)
    print("\n--- SINGLE URL: FETCH → SUMMARIZE → SAVE ---")
    print(res1.output)

    # 2) Multi-URL indexing in one prompt
    res2 = await agent.run(EXAMPLE_BATCH, deps=deps)
    print("\n--- MULTI-URL INDEXING RESULT ---")
    print(res2.output)

    # 3) Research-style question → RAG first → web fallback with citations
    res3 = await agent.run(EXAMPLE_QUESTION, deps=deps)
    print("\n--- QUESTION (RAG-FIRST WITH CITATIONS) ---")
    print(res3.output)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
