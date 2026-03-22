from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings

def risk_agent(state: ResearchState,llm) ->ResearchState:
    print(f"Risk Agent Running for: {state['company']}")

    try:

        risk_analysis = llm.invoke(f"""
        You are a senior financial risk analyst.
        Analyze the following data about {state['company']} 
        and identify key risks.
        
        News Data:
        {state['news']}
        
        Financial Data:
        {state['financial_data']}
        
        Provide:
        1. Top 3 risks with severity (High/Medium/Low)
        2. Risk score overall (1-10, 10 = highest risk)
        3. One sentence recommendation (Buy/Hold/Avoid)
        
        Be direct and specific.
        """)

        state['risk_assessment'] = risk_analysis.content
        state['status'] = "risk_analysis_complete"

        db = SessionLocal()
        log_agent(db, state['session_id'], "risk_agent", "completed", state['risk_assessment'])
        db.close()

        print("Risk Agent Complete")


    except Exception as e:
        state['errors'].append(f"Risk Agent Failed: Error {str(e)}")
        state['risk_assessment'] = "Risk Data Unavailable"
        state['status'] = "risk_agent_failed"

    return state


if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from database.connection import create_tables
    
    create_tables()
    
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0
    )
    
    test_state = {
        "company": "Infosys",
        "news": None,
        "financial_data": None,
        "risk_assessment": None,
        "critic_data": None,
        "final_report": None,
        "document_analysis": None,
        "errors": [],
        "status": "started",
        "session_id": "test-123",
        "confidence_score": None
    }
    
    result = risk_agent(test_state, llm)
    print(result["risk_assessment"])