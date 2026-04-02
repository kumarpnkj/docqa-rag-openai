import os
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def load_pdfs(folder="docs"):
    docs = []
    for pdf in Path(folder).glob("*.pdf"):
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    return docs

def main():
    load_dotenv()
    os.makedirs("vectorstore", exist_ok=True)

    print("Loading PDFs...")
    docs = load_pdfs("docs")
    print(f"Loaded {len(docs)} pages")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings()
    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local("vectorstore")
    print("Vector store saved to ./vectorstore")

if __name__ == "__main__":
    main()
