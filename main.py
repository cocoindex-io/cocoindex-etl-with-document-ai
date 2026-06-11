"""
CocoIndex ETL with Google Document AI (v1).

Pipeline: walk local PDFs -> parse to text with Google Document AI -> chunk ->
embed -> store in Postgres/pgvector. Re-runs only reprocess changed files, and
the (paid) Document AI parse is memoized by file content.

Index:
    cocoindex update main.py
    cocoindex update -L main.py      # live mode (watch the source dir)

Query the index:
    python main.py "your query"

Docs: https://cloud.google.com/document-ai/docs/try-docai
"""

from __future__ import annotations

import asyncio
import functools
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Annotated, AsyncIterator

import asyncpg
from dotenv import load_dotenv
from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from numpy.typing import NDArray
from pgvector.asyncpg import register_vector

import cocoindex as coco
from cocoindex.connectors import localfs, postgres
from cocoindex.ops.sentence_transformers import SentenceTransformerEmbedder
from cocoindex.ops.text import RecursiveSplitter
from cocoindex.resources.chunk import Chunk
from cocoindex.resources.file import FileLike, PatternFilePathMatcher
from cocoindex.resources.id import IdGenerator

TABLE_NAME = "document_ai_embeddings"
PG_SCHEMA_NAME = "coco_examples"
TOP_K = 5

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PG_DB = coco.ContextKey[asyncpg.Pool]("document_ai_db")
EMBEDDER = coco.ContextKey[SentenceTransformerEmbedder]("embedder", detect_change=True)

_splitter = RecursiveSplitter()


@functools.cache
def _docai() -> tuple[documentai.DocumentProcessorServiceAsyncClient, str]:
    """Lazily build the Document AI async client and resolve the processor path.

    Supported file types: https://cloud.google.com/document-ai/docs/file-types
    """
    project_id = os.environ["GOOGLE_CLOUD_PROJECT_ID"]
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us")
    processor_id = os.environ["GOOGLE_CLOUD_PROCESSOR_ID"]

    opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceAsyncClient(client_options=opts)
    return client, client.processor_path(project_id, location, processor_id)


@coco.fn(memo=True)
async def pdf_to_markdown(content: bytes) -> str:
    client, processor_name = _docai()
    response = await client.process_document(
        request=documentai.ProcessRequest(
            name=processor_name,
            raw_document=documentai.RawDocument(
                content=content, mime_type="application/pdf"
            ),
        )
    )
    return response.document.text


@dataclass
class DocumentEmbedding:
    id: int
    filename: str
    chunk_start: int
    chunk_end: int
    text: str
    embedding: Annotated[NDArray, EMBEDDER]


@coco.lifespan
async def coco_lifespan(builder: coco.EnvironmentBuilder) -> AsyncIterator[None]:
    database_url = os.getenv("POSTGRES_URL")
    if not database_url:
        raise ValueError("POSTGRES_URL is not set")

    async with asyncpg.create_pool(database_url) as pool:
        builder.provide(PG_DB, pool)
        builder.provide(EMBEDDER, SentenceTransformerEmbedder(EMBED_MODEL))
        yield


@coco.fn
async def process_chunk(
    chunk: Chunk,
    filename: pathlib.PurePath,
    id_gen: IdGenerator,
    table: postgres.TableTarget[DocumentEmbedding],
) -> None:
    table.declare_row(
        row=DocumentEmbedding(
            id=await id_gen.next_id(chunk.text),
            filename=str(filename),
            chunk_start=chunk.start.char_offset,
            chunk_end=chunk.end.char_offset,
            text=chunk.text,
            embedding=await coco.use_context(EMBEDDER).embed(chunk.text),
        ),
    )


@coco.fn(memo=True)
async def process_file(
    file: FileLike,
    table: postgres.TableTarget[DocumentEmbedding],
) -> None:
    markdown = await pdf_to_markdown(await file.read())
    chunks = _splitter.split(
        markdown, chunk_size=2000, chunk_overlap=500, language="markdown"
    )
    id_gen = IdGenerator()
    await coco.map(process_chunk, chunks, file.file_path.path, id_gen, table)


@coco.fn
async def app_main(sourcedir: pathlib.Path) -> None:
    target_table = await postgres.mount_table_target(
        PG_DB,
        table_name=TABLE_NAME,
        table_schema=await postgres.TableSchema.from_class(
            DocumentEmbedding,
            primary_key=["id"],
        ),
        pg_schema_name=PG_SCHEMA_NAME,
    )

    files = localfs.walk_dir(
        sourcedir,
        recursive=True,
        path_matcher=PatternFilePathMatcher(included_patterns=["**/*.pdf"]),
        live=True,  # pass -L to `cocoindex update` to actually run live
    )
    await coco.mount_each(process_file, files.items(), target_table)


app = coco.App(
    coco.AppConfig(name="DocumentAiEmbedding"),
    app_main,
    sourcedir=pathlib.Path("./pdf_files"),
)


# ============================================================================
# Query demo
# ============================================================================


async def query_once(
    pool: asyncpg.Pool,
    embedder: SentenceTransformerEmbedder,
    query: str,
    *,
    top_k: int = TOP_K,
) -> None:
    query_vec = await embedder.embed(query)
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT filename, text, embedding <=> $1 AS distance
            FROM "{PG_SCHEMA_NAME}"."{TABLE_NAME}"
            ORDER BY distance ASC
            LIMIT $2
            """,
            query_vec,
            top_k,
        )

    for r in rows:
        score = 1.0 - float(r["distance"])
        print(f"[{score:.3f}] {r['filename']}")
        print(f"    {r['text']}")
        print("---")


async def query(initial_query: str | None = None) -> None:
    database_url = os.getenv("POSTGRES_URL")
    if not database_url:
        raise ValueError("POSTGRES_URL is not set")

    embedder = SentenceTransformerEmbedder(EMBED_MODEL)
    async with asyncpg.create_pool(database_url, init=register_vector) as pool:
        if initial_query is not None:
            await query_once(pool, embedder, initial_query)
            return

        while True:
            q = input("Enter search query (or Enter to quit): ").strip()
            if not q:
                break
            await query_once(pool, embedder, q)


if __name__ == "__main__":
    load_dotenv()
    initial = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    asyncio.run(query(initial))
