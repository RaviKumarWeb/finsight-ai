from datetime import date
from graph.state import ResearchState
from database.crud import log_agent, update_session
from database.connection import SessionLocal
from config.settings import settings

def report_agent(state: ResearchState, llm) -> ResearchState:
    print(f"Report Agent Running for: {state['company']}")
    
    state['errors'] = state.get('errors', [])
    session_id = state.get('session_id', 'unknown')
    retry_count = state.get('retry_count', 0)

    # 1. Identify if this is a correction or a first draft
    is_retry = state.get('status') == "critic_flagged"
    critic_feedback = state.get('critic_data', '') if is_retry else ""

    # Build the prompt dynamically
    instruction = "Write a professional investment report" if not is_retry else "FIX AND RE-WRITE the previous report based on critic feedback"
    feedback_section = f"\n[CRITIC FEEDBACK - FIX THESE ERRORS]:\n{critic_feedback}" if is_retry else ""

    try:
        # 2. Invoke LLM with Context + Optional Feedback
        settings.increment_calls("report_agent")
        report = llm.invoke(f"""
        You are an Expert Financial Analyst. 
        TASK: {instruction} (Attempt: {retry_count + 1})

        [INPUT DATA]
        - News Summary: {state.get('news', 'No news available')}
        - Financial Metrics: {state.get('financial_data', 'No financial data available')}
        - Risk Assessment: {state.get('risk_assessment', 'No risk analysis available')}
        {feedback_section}

        [REPORT STRUCTURE]
        # {state['company']} Research Report
        ## Date: {date.today()}
        
        ## Executive Summary
        (3-4 concise lines summarizing the current situation)

        ## Key News & Developments
        (Bullet points of recent major events)

        ## Financial Performance
        (Analysis of key numbers, growth, and currency context)

        ## Risk Analysis
        (List risks with severity: High/Medium/Low)

        ## Investment Recommendation
        (Explicitly state Buy, Hold, or Avoid with a 1-sentence reason)
        
        Ensure all numbers match the 'Financial Metrics' provided.
        """)

        state['final_report'] = report.content
        state['status'] = "report_complete"

        # 3. Log and Update Database
        db = SessionLocal()
        try:
            log_agent(db, session_id, "report_agent", "completed", state['final_report'])
            update_session(db, session_id, final_report=state['final_report'])
        finally:
            db.close()

        print(f"Report Agent Complete for {state['company']} (Retry: {retry_count})")

    except Exception as e:
        error_msg = f"Report Agent Failed: {str(e)}"
        state['errors'].append(error_msg)
        state['final_report'] = "Final Report Generation Failed."
        state['status'] = "report_agent_failed"
        
        print(f"Error in Report Agent: {error_msg}")

        db = SessionLocal()
        try:
            log_agent(db, session_id, "report_agent", "failed", error_msg)
            update_session(db, session_id, final_report=state['final_report'])
        finally:
            db.close()

    return state


if __name__ == "__main__":
    from langchain_groq import ChatGroq
    from database.connection import create_tables
    import time

    create_tables()

    llm_test = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=settings.GROQ_API_KEY,
        temperature=0
    )

    # Example Test State simulating a retry
    test_state = {
        "company": "Infosys",
        "news": "Revenue growth of 9% reported.",
        "financial_data": "Revenue: $19.8B, PE: 17.0",
        "risk_assessment": "Recommendation: Hold",
        "critic_data": "The previous report said revenue was $50B, but it is actually $19.8B.",
        "status": "critic_flagged", # Simulating that it was flagged
        "retry_count": 1,
        "session_id": f"test-retry-{int(time.time())}",
        "errors": []
    }

    result = report_agent(test_state, llm_test)
    print("\n--- REPORT OUTPUT ---")
    print(result["final_report"])
