import yfinance as yf
from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from qdrant_client import models
from typing import List
load_dotenv()

MODEL_ID = "models/text-embedding-004"
embeddings = GoogleGenerativeAIEmbeddings(model=MODEL_ID)
DATASTORE_URI = "./datastore/vstore.db"
client = QdrantClient(path=DATASTORE_URI)


def format_document_for_llm(doc):
    metadata_str = ""
    if doc.metadata:
        relevant_fields = [
            "source",
            "title",
            "date",
            "author",
            "section",
            "page",
            "document_type",
        ]
        metadata_items = []
        for field in relevant_fields:
            if field in doc.metadata:
                metadata_items.append(f"{field}: {doc.metadata[field]}")

        if metadata_items:
            metadata_str = f"[{', '.join(metadata_items)}]\n\n"

    return f"{metadata_str}{doc.page_content}"

class RetrieveDocumentsInput(BaseModel):
    query: str = Field(description="The query to search for in the documents")
    collection: str = Field(description="A collection name to search within")
    document_sources: List[str] = Field(description="A list of document sources to filter the search")

@tool(args_schema=RetrieveDocumentsInput)
def retrieve_from_documents(query: str, collection: str, document_sources: List[str]) -> str:
    """
    Retrieve relevant information from a list of documents based on the query.

    Returns:
        Formatted documents containing the relevant information.
    """
    print("tool called with query:", query)
    try:
        query_vector = embeddings.embed_query(query)
        k = 10
        qdrant = QdrantVectorStore(
            client=client,
            collection_name="documents/" + collection,
            embedding=embeddings,
            retrieval_mode=RetrievalMode.DENSE,
        )
        found_docs = qdrant.similarity_search_by_vector(
            embedding=query_vector, 
            k=k, 
            
        )
        print(f"Found {len(found_docs)} documents for query: {query}")    
        
        if not found_docs:
            return "No relevant documents found for the given query and sources."

        context = "\n\n".join([format_document_for_llm(doc) for doc in found_docs])
        return context
        
    except Exception as e:
        print(f"Error retrieving documents: {str(e)}")
        return f"Error retrieving documents: {str(e)}"

@tool
def yfinance_get_dividends_stock_split_history(symbol: str) -> str:
    """Get the dividend and stock split history for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.acitons.to_dict()

    return response


@tool
def yfinance_get_balance_sheet(symbol: str) -> str:
    """Get the balance sheet for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.balance_sheet.to_dict()

    return response


@tool
def yfinance_get_financials(symbol: str) -> str:
    """Get the financials for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.financials.to_dict()

    return response


@tool
def yfinance_get_earnings(symbol: str) -> str:
    """Get the earnings for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.earnings.to_dict()

    return response


@tool
def yfinance_get_info(symbol: str) -> str:
    """Get the information for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.info

    return response


@tool
def yfinance_get_cash_flow(symbol: str) -> str:
    """Get the cash flow for a given stock symbol."""
    data = yf.Ticker(symbol)
    response = data.cashflow.to_dict()

    return response


financial_agent_tools = [
    yfinance_get_dividends_stock_split_history,
    yfinance_get_balance_sheet,
    yfinance_get_financials,
    yfinance_get_earnings,
    yfinance_get_info,
    yfinance_get_cash_flow
]
doc_agent_tools = [
    retrieve_from_documents
]
