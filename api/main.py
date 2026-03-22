import uuid
import asyncio
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from config.settings import settings
from langchain_groq import ChatGroq
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
from database.connection import SessionLocal, create_tables
from graph.research_graph import build_graph
from database.crud import create_session, update_session,get_session,get_company_history, log_agent

executor = ThreadPoolExecutor(max_workers=3)
app = FastAPI()
create_tables()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=settings.GROQ_API_KEY,
    temperature=0
)

graph = build_graph(llm)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ResearchRequest(BaseModel):
    company: str


@app.post("/research")
async def research_company(request: ResearchRequest):
    session_id = str(uuid.uuid4())
    
    db = SessionLocal()
    create_session(db, session_id, request.company)
    db.close()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: graph.invoke(
                {
                    "company": request.company,
                    "news": None,
                    "financial_data": None,
                    "risk_assessment": None,
                    "critic_data": None,
                    "final_report": None,
                    "document_analysis": None,
                    "errors": [],
                    "status": "started",
                    "session_id": session_id,
                    "confidence_score": None,
                },
                config={"recursion_limit": 25}
            )
        )
        return {
            "session_id": session_id,
            "status": "success",
            "report": result.get('final_report'),
            "confidence": result.get('confidence_score'),
            "warning": "Low confidence — verify independently" 
               if result.get('confidence_score') and result.get('confidence_score') < 0.7 
               else None,
            "verdict": result.get('risk_assessment', '')[:50],
            "errors": result.get('errors', [])
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "status": "error",
            "error": str(e)
        }

@app.get("/report/{session_id}")
def get_report(session_id: str, db: Session = Depends(get_db)):
    session = get_session(db,session_id)
    if not session:
        return {"error": "Session not found", "id": session_id}
    
    return {
        "session_id": session.id,
        "status": session.status,
        "data": session
    }


@app.get("/health")
def check_health():
    return {"status": "healthy", "message": "The Research API is running"}
