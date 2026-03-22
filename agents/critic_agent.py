from graph.state import ResearchState
from database.crud import log_agent, update_session
from database.connection import SessionLocal
from config.settings import settings

def extract_verdict(state):
    # Extract Buy/Hold/Avoid from risk assessment
    risk = state.get('risk_assessment', '') or ''
    if 'Buy' in risk: return 'BUY'
    if 'Avoid' in risk: return 'AVOID'
    return 'HOLD'

def critic_agent(state: ResearchState,llm) -> ResearchState:
    print(f"Critic Agent Runnig for: {state['company']}")

    try:
        critic_summary = llm.invoke(f"""
        You are a financial fact-checker.
        REPORT TO CHECK: {state['final_report']}
        ORIGINAL DATA: {state['financial_data']}

        Compare the report against the original data.
        List mismatches, give a confidence score (0.0 to 1.0), and a verdict.
        
        Format:
        MISMATCHES: [list]
        CONFIDENCE: [number]
        VERDICT: [Approved/Needs Revision]
        """)

        # After getting LLM response
        response_text = critic_summary.content
        confidence = 0.8  # default to approved if parsing fails

        for line in response_text.split('\n'):
            if 'CONFIDENCE:' in line:
                try:
                    # Remove any markdown formatting
                    clean = line.split(':')[1].strip()
                    clean = clean.replace('*', '').replace('_', '').strip()
                    confidence = float(clean)
                    print(f"Confidence in Try: {confidence}")
                    break
                except:
                    confidence = 0.8  # default to approved
        
        state['critic_data'] = response_text
        state['confidence_score'] = confidence

        if state['confidence_score'] > 0.8:
            state['status'] = "critic_approved"
        else:
            state['status'] = "critic_flagged"

        db = SessionLocal()
        log_agent(db, state['session_id'], "critic_agent", "completed",state['critic_data'])
        update_session(db,state['session_id'],confidence_score=state['confidence_score'],status=state['status'],final_report=state['final_report'],verdict=extract_verdict(state))
        db.close()

        print("Critic Agent Complete")
        
    except Exception as e:
        state['errors'].append(f"Critic Agent Failed: Error {str(e)}")
        state['critic_data'] = "Critic Data Unavaliable"
        state['status'] = "critic_agent_failed"

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
    "news": "Infosys Q3 revenue $5.1 billion, growth 1.7%",
    "financial_data": "Revenue: $19.8B, EPS: $0.80, PE: 17.78",
    "risk_assessment": "Risk 6/10, Recommendation: Hold",
    "critic_data": None,
    "final_report": "Infosys revenue was $20B with EPS of $1.50",
    # ↑ deliberately wrong number — critic should catch this
    "document_analysis": None,
    "errors": [],
    "status": "started",
    "session_id": "test-123",
    "confidence_score": None
    }
    
    result = critic_agent(test_state, llm)
    print(f"{result['critic_data']}\n Confidence Score {result['confidence_score']}")