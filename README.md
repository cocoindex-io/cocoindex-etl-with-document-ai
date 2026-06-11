# 🥥 CocoIndex ETL with Document AI

[CocoIndex](https://cocoindex.io) is an ETL framework to transform data for AI, with real-time incremental processing — keep your index up to date with low latency on source updates. It supports custom logic like LEGO, making it easy to plug in the modules that best suit your project.

This example builds an embedding index from local PDF files, using [Google Document AI](https://cloud.google.com/document-ai?hl=en) as the parser. The pipeline walks a folder of PDFs, parses each one with Document AI, chunks the text, embeds it, and stores everything in Postgres/pgvector — then lets you query it semantically.

🥥 🌴 We are constantly improving — more blogs and examples coming soon. Stay tuned 👀 and **drop a star at [CocoIndex on GitHub](https://github.com/cocoindex-io/cocoindex)** for the latest updates!
[![GitHub](https://img.shields.io/github/stars/cocoindex-io/cocoindex?color=5B5BD6)](https://github.com/cocoindex-io/cocoindex)

![Use Document AI to parse PDF files in CocoIndex](image/coco-doc-ai.png)

## Prerequisites

- **Python 3.11+**
- **Postgres with pgvector** — [install Postgres](https://cocoindex.io/docs/getting_started/installation#-install-postgres) if you don't have one.
- **A Google Document AI processor:**
    - Follow the [official Document AI quickstart](https://cloud.google.com/document-ai/docs/try-docai).
    - Sign in to the [Google Cloud Console](https://console.cloud.google.com/), create or open a project, and enable the Document AI API.
      ![Google Cloud Console](image/cloud-console.png)
    - Create a processor in Document AI.
      ![Create a processor in Document AI](image/create-processor.png)
    - Authenticate locally so the client can pick up credentials: `gcloud auth application-default login`.
- Fill in `.env` with your `GOOGLE_CLOUD_PROJECT_ID`, `GOOGLE_CLOUD_PROCESSOR_ID`, and (if not `us`) `GOOGLE_CLOUD_LOCATION`. `POSTGRES_URL` points at your pgvector database.

## Run

Install dependencies:

```bash
pip install -e .
```

Build the index — parse each PDF with Document AI, chunk, embed, and write to Postgres:

```bash
cocoindex update main.py
```

Run it incrementally as files change (live mode watches the source dir):

```bash
cocoindex update -L main.py
```

Re-runs only reprocess files that actually changed, and the (paid) Document AI parse is memoized by file content — so editing chunking or embedding settings won't re-bill the API for unchanged PDFs.

## Query

Run a one-off semantic search:

```bash
python main.py "what is multi-head attention?"
```

Or start an interactive prompt:

```bash
python main.py
```

## CocoInsight

CocoInsight is in early access (free 😊). A quick 3-minute tutorial: [Watch on YouTube](https://youtu.be/ZnmyoHslBSc?si=pPLXWALztkA710r9).

Explore your pipeline — sources, transformations, and lineage — in the CocoInsight UI:

```bash
cocoindex server -ci main.py
```

Then open [https://cocoindex.io/cocoinsight](https://cocoindex.io/cocoinsight).
