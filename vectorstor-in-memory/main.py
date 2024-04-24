from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



if __name__ == '__main__':
    print("hi")
    pdf_path = "/Users/admin/LLM-study/vectorstor-in-memory/2210.03629.pdf"
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    res = qa.run("Give me the gist of ReAct in 3 sentences")

    print(res)



