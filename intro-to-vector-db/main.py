import os

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore


if __name__ == '__main__':
    print("Hello VectorStore!")
    loader = TextLoader("/Users/admin/LLM-study/intro-to-vector-db/mediumblogs/mediumblog1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = PineconeVectorStore.from_documents(texts, embeddings, index_name="medium-blogs-embeddings-index")

    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0.0
    )

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever())

    query = "What is a vector DB? Give me a 15 word answer for a begginner"

    result = qa({"query": query})
    print(result)