import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_docling import DoclingLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_ID = "models/text-embedding-004"
embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_ID)
root = Path("./documents")

DATASTORE_URI = "./datastore/vstore.db"
client = QdrantClient(path=DATASTORE_URI)


if not root.exists():
    raise Exception(f"Directory {root} not found")

pdfs = {}
for directory in root.iterdir():
    if directory.is_dir():
        pdfs[directory.as_posix()] = list()
        company_transcripts = []
        for file in directory.iterdir():
            if file.suffix in [".pdf"]:
                company_transcripts.append(file)
        pdfs[directory.as_posix()].extend(company_transcripts)

logger.info(f"Found {len(pdfs)} PDF files to process")

for pdf in tqdm(pdfs, desc="Processing PDFs", unit="file"):
    collection_name = pdf
    collection = pdfs[pdf]
    client.create_collection(
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        collection_name=collection_name,
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
    for file in tqdm(
        collection,
        desc=f"Processing files in {collection_name}",
        unit="file",
        leave=False,
    ):
        logger.info(f"Starting processing of: {file.name}")

        file_path = str(file.resolve())
        logger.info(f"Loading document from: {file_path}")
        loader = DoclingLoader(
            file_path=file_path,
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document chunks")

        for document in documents:
            document.metadata["source"] = file.as_posix()
        document.metadata["collection_name"] = collection_name

        logger.info(f"Adding documents to QdrantClient collection: {collection_name}")
        vector_store.add_documents(documents)
        logger.info(f"Successfully processed and stored: {file.name}")

logger.info("All PDF files have been processed successfully")
