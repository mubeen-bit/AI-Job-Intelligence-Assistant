import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_core.documents import Document
from dotenv import load_dotenv







MODEL = "gpt-4.1-nano"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

load_dotenv(override=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")



 
def fetch_documents():
    """Load all markdown documents"""

    documents = []

    for file in Path(KNOWLEDGE_BASE_PATH).rglob("*.md"):

        with open(file, "r", encoding="utf-8") as f:

            text = f.read()

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "type": "job",
                        "source": file.as_posix(),
                    },
                )
            )

    print(f"Loaded {len(documents)} documents")

    return documents
def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")
