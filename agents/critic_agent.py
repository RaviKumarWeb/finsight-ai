import re
from graph.state import ResearchState
from database.crud import log_agent, update_session
from database.connection import SessionLocal
from config.settings import settings

def extract_verdict(state):
    risk = str(state.get('risk_assessment', ''))
    report = str(state.get('final_report', ''))
    combined = (risk + report).upper()
    
    # Count occurrences to determine strongest signal
    buy_signals = combined.count('STRONG BUY') * 3 + combined.count('**BUY**') * 2 + combined.count('RECOMMENDATION: BUY')
    avoid_signals = combined.count('STRONG AVOID') * 3 + combined.count('**AVOID**') * 2 + combined.count('RECOMMENDATION: AVOID')
    hold_signals = combined.count('**HOLD**') * 2 + combined.count('RECOMMENDATION: HOLD')
    
    if buy_signals > avoid_signals and buy_signals > hold_signals:
        return 'BUY'
    if avoid_signals > buy_signals and avoid_signals > hold_signals:
        return 'AVOID'
    return 'HOLD'

def critic_agent(state: ResearchState, llm) -> ResearchState:
    print(f"Critic Agent Running for: {state['company']}")
    
    # 1. Initialize essential state keys
    state['errors'] = state.get('errors', [])
    session_id = state.get('session_id', 'unknown')
    
    # Get current retry count
    current_retry = state.get('retry_count', 0)

    try:
        # 2. Fact-check the report against original data
        settings.increment_calls("critic_agent")
        critic_summary = llm.invoke(f"""
        You are a Financial Fact-Checker. 
        Compare the Research Report against the Original Data provided.
        
        [ORIGINAL DATA]
        {state.get('financial_data', 'No data available')}
        
        [RESEARCH REPORT]
        {state.get('final_report', 'No report available')}

        Identify any mismatches in numbers, dates, or company names.
        
        Format your response EXACTLY like this:
        MISMATCHES: [List specific errors or say 'None']
        CONFIDENCE: [A number between 0.0 and 1.0 based on report accuracy]
        VERDICT: [Approved or Needs Revision]
        """)

        response_text = critic_summary.content
        state['critic_data'] = response_text
        
        # 3. Robust Parsing for Confidence Score
        confidence = 0.8  # Default safety value
        try:
            # Look for "CONFIDENCE: 0.x" anywhere in the response
            match = re.search(r"CONFIDENCE:\s*([\d\.]+)", response_text)
            if match:
                confidence = float(match.group(1))
                print(f"Parsed Confidence: {confidence}")
        except Exception as parse_err:
            print(f"Confidence parsing failed: {parse_err}")

        state['confidence_score'] = confidence

        # 4. Determine Status and increment Retry Count logic
        # Logic: If confidence is low, we only flag for a re-run if we haven't hit the limit
        if confidence >= 0.8:
            state['status'] = "critic_approved"
        else:
            if current_retry >= 2:
                # Force exit to prevent infinite loop / rate limits
                state['status'] = "critic_approved" 
                state['errors'].append("Max retries (2) reached. Ending with current report version.")
            else:
                state['status'] = "critic_flagged"
                # INCREMENT: This informs the graph to loop back
                state['retry_count'] = current_retry + 1

        # 5. Update Database
        db = SessionLocal()
        try:
            # Log the specific agent step
            log_agent(db, session_id, "critic_agent", "completed", state['critic_data'])
            
            # Finalize the session data
            update_session(
                db, 
                session_id, 
                confidence_score=state['confidence_score'],
                status=state['status'],
                final_report=state.get('final_report'),
                verdict=extract_verdict(state)
            )
        finally:
            db.close()

        print(f"Critic Agent Complete: {state['status']} (Next Retry Count: {state.get('retry_count', current_retry)})")

    except Exception as e:
        error_msg = f"Critic Agent Failed: {str(e)}"
        state['errors'].append(error_msg)
        state['critic_data'] = "Critic Analysis Unavailable"
        state['status'] = "critic_agent_failed"
        
        print(f"Error in Critic Agent: {error_msg}")
        
        db = SessionLocal()
        try:
            log_agent(db, session_id, "critic_agent", "failed", error_msg)
        finally:
            db.close()

    return state
