from datetime import date
from graph.state import ResearchState
from database.crud import log_agent, update_session
from database.connection import SessionLocal
from config.settings import settings

def report_agent(state: ResearchState,llm) ->ResearchState:
    print(f"Report Agent Running for: {state['company']}")

    try:
        report = llm.invoke(f"""
        You are an expert financial analyst. 
        Use the following gathered information to write a professional report:

        DATA PROVIDED:
        - News: {state['news']}
        - Financials: {state['financial_data']}
        - Risks: {state['risk_assessment']}

        STRUCTURE:
        # {state['company']} Research Report
        ## Date: {date.today()}
        ## Executive Summary
        (3-4 concise lines summarizing the company's current standing)

        ## Key News & Developments
        (Bullet points highlighting recent events)

        ## Financial Performance
        (Key numbers and growth metrics)

        ## Risk Analysis
        (List risks with a severity level: High/Medium/Low)

        ## Investment Recommendation
        (Explicitly state Buy, Hold, or Avoid with a clear reason)
        """)
        state['final_report'] = report.content
        state['status'] = "report_complete"

        db = SessionLocal()
        log_agent(db, state['session_id'], "report_agent", "completed",state['final_report'])
        update_session(db, state['session_id'], final_report=state['final_report'])
        db.close()

        print("Report Agent Complete")
    except Exception as e:
        state['errors'].append(f"Report Agent Failed: Error {str(e)}")
        state['final_report'] = "Final Report Data Unavaliable"
        state['status'] = "report_agent_failed"
    return state


if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from database.connection import create_tables
    from agents.news_agent import news_agent
    from agents.financial_agent import financial_agent
    from agents.risk_agent import risk_agent

    create_tables()

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0
    )

    test_state = {
        "company": "Tesla",
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

    # Run all agents in order
    test_state = news_agent(test_state, llm)
    test_state = financial_agent(test_state, llm)
    test_state = risk_agent(test_state, llm)
    result = report_agent(test_state, llm)
    
    print("\n--- FINAL REPORT ---")
    print(result["final_report"])