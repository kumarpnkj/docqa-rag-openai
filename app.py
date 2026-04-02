import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="DocQA RAG", layout="wide")
st.title("📚 DocQA RAG — Ask your PDFs")

SYSTEM = """You are a helpful assistant. Use the provided context to answer.
If unsure, say you don't know. Always cite sources as (source, page)."""

PROMPT_TMPL = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

@st.cache_resource
def load_vs():
    embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

def format_context(docs):
    out = []
    for d in docs:
        meta = d.metadata
        src = meta.get("source", "unknown")
        page = meta.get("page", "?")
        out.append(f"[{src} | p.{page+1}]\n{d.page_content}")
    return "\n\n---\n\n".join(out)

vs = load_vs()
question = st.text_input("Ask a question about your PDFs:")

if question.strip():
    docs = vs.similarity_search(question, k=4)
    context = format_context(docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
    msg = PROMPT_TMPL.format_messages(question=question, context=context)
    resp = llm.invoke(msg).content

    st.markdown("### Answer")
    st.write(resp)
    st.markdown("### Sources")
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "?")+1
        st.write(f"- **{src}**, p.{page}")
