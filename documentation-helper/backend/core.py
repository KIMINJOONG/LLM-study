from typing import Any, List, Dict

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.vectorstores.pinecone import Pinecone

from consts import INDEX_NAME


def run_llm(query: str, chat_history: List[Dict[str, Any]]) -> Any:
    embeddings = OpenAIEmbeddings()
    dosearch = Pinecone.from_existing_index(index_name=INDEX_NAME, embedding=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=dosearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"question": query, "chat_history": chat_history})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))
