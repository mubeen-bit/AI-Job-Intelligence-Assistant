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

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def fetch_documents():
    documents = []
    for file in Path(KNOWLEDGE_BASE_PATH).rglob("*.md"):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        # Extract job title once at load time
        job_title = ""
        for line in text.split("\n"):
            if line.startswith("# Job Title:"):
                job_title = line.replace("# Job Title:", "").strip()
                break

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "type": "job",
                    "source": file.as_posix(),
                    "job_title": job_title,  # ← reliable from the start
                },
            )
        )

    print(f"Loaded {len(documents)} documents")
    return documents
def create_chunks(documents):
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n# ",        # Job Title header
            "\n## ",       # Section headers (Description, Job ID)
            "\n---",       # The divider between JDs
            "\n\n",        # Paragraph breaks
            "\n",          # Line breaks (skills are one per line)
        ],
        chunk_size=1000,
        chunk_overlap=150,
        keep_separator=True,
    )
    chunks = text_splitter.split_documents(documents)

    # Inject job title into every chunk so it's always searchable
    for chunk in chunks:
        lines = chunk.page_content
        job_title = chunk.metadata.get("job_title", "")
        
        # Extract from content if not in metadata
        if not job_title:
            for line in lines.split("\n"):
                if "Job Title:" in line:
                    job_title = line.replace("# Job Title:", "").strip()
                    break
        
        # Prepend title if not already there
        if job_title and f"Job Title: {job_title}" not in chunk.page_content:
            chunk.page_content = f"Job Title: {job_title}\n\n{chunk.page_content}"
            chunk.metadata["job_title"] = job_title

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

    # Sanity check before wiping your DB
    print(f"Total chunks: {len(chunks)}")
    print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")
    print(f"Sample chunk:\n{chunks[0].page_content}")
    print(f"Metadata: {chunks[0].metadata}")

    create_embeddings(chunks)
    print("Ingestion complete")
