import os
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone
from langchain_openai.embeddings import OpenAIEmbeddings

from consts import INDEX_NAME


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="/Users/admin/LLM-study/documentation-helper/langchain-docs/langchain.readthedocs.io/en/latest"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    text_spliter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_spliter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")

    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("******* Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
