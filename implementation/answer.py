from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
DB_NAME = str(Path(__file__).parent.parent / "vector_db")

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
RETRIEVAL_K = 20

# 2. Your system prompt is good but add a student-focused instruction
SYSTEM_PROMPT = """
You are an AI Job Intelligence Assistant helping university students understand the job market.

You help users understand:
- job descriptions and what they actually mean
- required skills and how to develop them
- technologies and tools used in industry
- what separates entry-level from mid-level roles
- career paths in tech and data roles

Your responses should be accurate, concise, and easy for a student to act on.
If the answer is not in the context, say so clearly — do not make up information.

Context:
{context}
"""

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
# retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": RETRIEVAL_K, "fetch_k": 40, "lambda_mult": 0.7}
)
llm = ChatOpenAI(temperature=0, model_name=MODEL)

def expand_query(question: str) -> str:
    """Use LLM to rewrite the question in JD-friendly language."""
    messages = [
        SystemMessage(content="""Rewrite the user's question as a job description search query.
Use industry terminology, skill names, and keywords an employer would write.
Return only the rewritten query, nothing else."""),
        HumanMessage(content=question)
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def fetch_context(question: str, history: list[dict] = []) -> list[Document]:
    last_user = next(
        (m["content"] for m in reversed(history) if m["role"] == "user"), ""
    )
    query = f"{last_user}\n{question}".strip() if last_user else question
    expanded = expand_query(query)  # ← rewrite before retrieval
    return retriever.invoke(expanded)


# def fetch_context(question: str, history: list[dict] = []) -> list[Document]:
#     """
#     Retrieve relevant context documents for a question.
#     """
#     last_user = next(
#         (m["content"] for m in reversed(history) if m["role"] == "user"), ""
#     )
#     query = f"{last_user}\n{question}".strip() if last_user else question
#     return retriever.invoke(query)  # k is now set on the retriever itself


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine the current question with only the last assistant response for LLM context.
    """
    last_assistant = next(
        (m["content"] for m in reversed(history) if m["role"] == "assistant"), ""
    )
    if last_assistant:
        return f"Previous response: {last_assistant}\n\nCurrent question: {question}"
    return question



def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    docs = fetch_context(question, history)  # ← pass history here
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs
