from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings

def risk_agent(state: ResearchState, llm) -> ResearchState:
    print(f"Risk Agent Running for: {state['company']}")
    
    state['errors'] = state.get('errors', [])
    session_id = state.get('session_id', 'unknown')

    try:
        # Check if we have the necessary data to perform an analysis
        if not state.get('news') or not state.get('financial_data'):
            raise ValueError("Missing News or Financial data for risk assessment.")

        # 1. Perform Risk Analysis with LLM
        settings.increment_calls("risk_agent")
        risk_analysis = llm.invoke(f"""
        You are a Senior Financial Risk Analyst. 
        Analyze the following context for {state['company']} as of March 2026.
        
        [CONTEXT DATA]
        News Summary: {state['news']}
        Financial Metrics: {state['financial_data']}
        
        [REQUIREMENTS]
        Provide a structured report including:
        1. Top 3 Risks: Identify specific threats with severity (High/Medium/Low).
        2. Overall Risk Score: Scale of 1-10 (10 = extreme risk).
        3. Recommendation: One sentence (Buy/Hold/Avoid) based strictly on these risks.
        
        Be direct, professional, and specific to the data provided.
        """)

        state['risk_assessment'] = risk_analysis.content
        state['status'] = "risk_analysis_complete"

        # 2. Log Success to Database
        db = SessionLocal()
        try:
            log_agent(db, session_id, "risk_agent", "completed", state['risk_assessment'])
        finally:
            db.close()

        print(f"Risk Agent Complete for {state['company']}")

    except Exception as e:
        error_msg = f"Risk Agent Failed: {str(e)}"
        state['errors'].append(error_msg)
        state['risk_assessment'] = "Risk Assessment Unavailable due to processing error."
        state['status'] = "risk_agent_failed"
        
        print(f"Error in Risk Agent: {error_msg}")

        # Save the failure reason to the database
        db = SessionLocal()
        try:
            log_agent(db, session_id, "risk_agent", "failed", error_msg)
        finally:
            db.close()

    return state
