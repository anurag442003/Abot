import os
import glob
import shutil
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import PersistentClient
from dotenv import load_dotenv

load_dotenv(override=True)

DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
KNOWLEDGE_BASE = str(Path(__file__).parent.parent / "knowledge-base")
collection_name = "docs"  # ← must match answer.py

embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B", model_kwargs={"trust_remote_code": True})


def fetch_documents():
    folders = glob.glob(str(Path(KNOWLEDGE_BASE) / "*"))
    documents = []
    for folder in folders:
        doc_type = os.path.basename(folder)
        loader = DirectoryLoader(
            folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"}
        )
        folder_docs = loader.load()
        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)
    return documents


def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    return text_splitter.split_documents(documents)


def create_embeddings(chunks):
    # Clean wipe — prevents corruption and dimension mismatch
    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)

    chroma = PersistentClient(path=DB_NAME)
    collection = chroma.get_or_create_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    metas = [chunk.metadata for chunk in chunks]
    vectors = embeddings.embed_documents(texts)
    ids = [str(i) for i in range(len(chunks))]

    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metas)

    count = collection.count()
    dimensions = len(vectors[0])
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")


if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")