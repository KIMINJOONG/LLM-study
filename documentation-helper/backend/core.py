from typing import Any

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.pinecone import Pinecone

from consts import INDEX_NAME


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    dosearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=dosearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})

if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
