import tempfile
import os

from dotenv import load_dotenv
from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from psycopg_pool import ConnectionPool

import cocoindex

class ToMarkdown(cocoindex.op.FunctionSpec):
    """Convert a PDF to markdown using Google Document AI."""

@cocoindex.op.executor_class(cache=True, behavior_version=1)
class DocumentAIExecutor:
    """Executor for Google Document AI to parse files.
       Supported file types: https://cloud.google.com/document-ai/docs/file-types
    """

    spec: ToMarkdown
    _client: documentai.DocumentProcessorServiceClient
    _processor_name: str

    def prepare(self):
        # Initialize Document AI
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us")
        processor_id = os.environ.get("GOOGLE_CLOUD_PROCESSOR_ID")
        
        # You must set the api_endpoint if you use a location other than 'us', e.g.:
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self._client = documentai.DocumentProcessorServiceClient(client_options=opts)
        self._processor_name = self._client.processor_path(project_id, location, processor_id)

    async def __call__(self, content: bytes) -> str:
        # Create the document object
        document = documentai.Document(
            content=content,
            mime_type="application/pdf"
        )
        
        # Process the document
        request = documentai.ProcessRequest(
            name=self._processor_name,
            raw_document=documentai.RawDocument(content=content, mime_type="application/pdf")
        )
        
        response = self._client.process_document(request=request)
        document = response.document
        
        # Extract the text from the document
        text = document.text
        
        # Convert to markdown format
        # This is a simple conversion - you might want to enhance this based on your needs
        # by using document.pages, entities, etc. for more structured markdown
        return text


def text_to_embedding(text: cocoindex.DataSlice) -> cocoindex.DataSlice:
    """
    Embed the text using a SentenceTransformer model.
    """
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(
            model="sentence-transformers/all-MiniLM-L6-v2"))

@cocoindex.flow_def(name="DocumentAiPdfEmbedding")
def pdf_embedding_flow(flow_builder: cocoindex.FlowBuilder, data_scope: cocoindex.DataScope):
    """
    Define an example flow that embeds files into a vector database.
    """
    data_scope["documents"] = flow_builder.add_source(cocoindex.sources.LocalFile(path="pdf_files", binary=True))

    doc_embeddings = data_scope.add_collector()

    with data_scope["documents"].row() as doc:
        doc["markdown"] = doc["content"].transform(ToMarkdown())
        doc["chunks"] = doc["markdown"].transform(
            cocoindex.functions.SplitRecursively(),
            language="markdown", chunk_size=2000, chunk_overlap=500)

        with doc["chunks"].row() as chunk:
            chunk["embedding"] = chunk["text"].call(text_to_embedding)
            doc_embeddings.collect(id=cocoindex.GeneratedField.UUID,
                                   filename=doc["filename"], location=chunk["location"],
                                   text=chunk["text"], embedding=chunk["embedding"])

    doc_embeddings.export(
        "doc_embeddings",
        cocoindex.storages.Postgres(),
        primary_key_fields=["id"],
        vector_indexes=[
            cocoindex.VectorIndexDef(
                field_name="embedding",
                metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY)])

def search(pool: ConnectionPool, query: str, top_k: int = 5):
    # Get the table name, for the export target in the pdf_embedding_flow above.
    table_name = cocoindex.utils.get_target_storage_default_name(
        pdf_embedding_flow, "doc_embeddings"
    )
    # Evaluate the transform flow defined above with the input query, to get the embedding.
    query_vector = text_to_embedding.eval(query)
    # Run the query and get the results.
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT filename, text, embedding <=> %s::vector AS distance
                FROM {table_name} ORDER BY distance LIMIT %s
            """,
                (query_vector, top_k),
            )
            return [
                {"filename": row[0], "text": row[1], "score": 1.0 - row[2]}
                for row in cur.fetchall()
            ]

def _main():
    # Initialize the database connection pool.
    pool = ConnectionPool(os.getenv("COCOINDEX_DATABASE_URL"))
    # Run queries in a loop to demonstrate the query capabilities.
    while True:
        query = input("Enter search query (or Enter to quit): ")
        if query == "":
            break
        # Run the query function with the database connection pool and the query.
        results = search(pool, query)
        print("\nSearch results:")
        for result in results:
            print(f"[{result['score']:.3f}] {result['filename']}")
            print(f"    {result['text']}")
            print("---")
        print()

if __name__ == "__main__":
    load_dotenv()
    cocoindex.init()
    _main()
