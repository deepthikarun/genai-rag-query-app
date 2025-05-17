from fastapi import FastAPI, Query
from pydantic import BaseModel
from app.rag_chain import load_pdf_and_create_vectorstore, get_qa_chain

app = FastAPI(
    title="RAG PDF QA API",
    description="Ask questions about your local PDF using LangChain + FAISS",
    version="1.0",
)

# Load the vector store on startup
pdf_path = "data/en-allianz-se-annual-report-2024.pdf"
vectorstore = load_pdf_and_create_vectorstore(pdf_path)
qa_chain = get_qa_chain(vectorstore)

class QueryRequest(BaseModel):
    query: str

@app.post("/ask/")
def ask_question(request: QueryRequest):
    try:
        result = qa_chain.run(request.query)
        return {"query": request.query, "answer": result}
    except Exception as e:
        return {"error": str(e)}


