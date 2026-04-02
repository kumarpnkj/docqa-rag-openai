import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

load_dotenv()

SYSTEM = """You are a helpful assistant. Use the provided context to answer.
If unsure, say you don't know. Always cite sources as (source, page)."""

PROMPT_TMPL = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

def load_vs(path="vectorstore"):
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

def format_context(docs):
    out = []
    for d in docs:
        meta = d.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        out.append(f"[{src} | p.{page+1}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)

def answer(question, k=4, model="gpt-4o-mini"):
    vs = load_vs()
    docs = vs.similarity_search(question, k=k)
    context = format_context(docs)
    llm = ChatOpenAI(model=model, temperature=0)
    msg = PROMPT_TMPL.format_messages(question=question, context=context)
    resp = llm.invoke(msg)
    cites = [(d.metadata.get("source"), d.metadata.get("page", "?")+1) for d in docs]
    return resp.content, cites

if __name__ == "__main__":
    q = input("Ask a question: ")
    ans, cites = answer(q)
    print("\n=== Answer ===\n", ans)
    print("\n=== Sources ===")
    for s, p in cites:
        print(f"- {s}, p.{p}")
