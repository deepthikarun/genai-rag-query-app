import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI  # ✅ Use ChatOpenAI for Groq

# Load .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Load PDF & create FAISS index
def load_pdf_and_create_vectorstore(pdf_path: str, db_dir: str = "faiss_index") -> FAISS:
    if os.path.exists(db_dir):
        return FAISS.load_local(
            db_dir,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True
        )

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    vectorstore = FAISS.from_documents(docs, embedding_model)
    vectorstore.save_local(db_dir)
    return vectorstore

# QA Chain with Groq Chat LLM
def get_qa_chain(vectorstore: FAISS) -> RetrievalQA:
    llm = ChatOpenAI(  # ✅ ChatOpenAI uses /chat/completions, which Groq supports
        model_name="llama3-8b-8192",  # or "mixtral-8x7b-32768"
        temperature=0.7,
        max_tokens=512,
        openai_api_key=GROQ_API_KEY,
        openai_api_base="https://api.groq.com/openai/v1",
    )

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Test
if __name__ == "__main__":
    pdf_path = "data/en-allianz-se-annual-report-2024.pdf"
    db_dir = "faiss_index"

    try:
        vectorstore = load_pdf_and_create_vectorstore(pdf_path, db_dir)
        print("vector store created")
        qa_chain = get_qa_chain(vectorstore)
        
        question = "What is the main topic of the document?"
        result = qa_chain.invoke(question)

        print(f"Q: {question}\nA: {result['result']}")
    except Exception as e:
        print(f"Error: {e}")
