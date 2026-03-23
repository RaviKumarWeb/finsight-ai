import time
from langchain_tavily import TavilySearch
from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings

def news_agent(state: ResearchState, llm) -> ResearchState:
    print(f"News Agent Running for: {state['company']}")
    
    state['errors'] = state.get('errors', [])
    session_id = state.get('session_id', 'unknown')

    try:
        # 1. Search for latest 2026 news
        search = TavilySearch(max_results=5)
        search_query = f"{state['company']} latest news 2026 financial performance and business updates"
        results = search.invoke(search_query)

        # 2. Summarize with LLM
        summary_prompt = f"""
        Summarize the latest news for {state['company']} in 4-5 key bullet points.
        Focus strictly on 2026 financial impact, business performance, and major deals.

        News Data: {results}
        """
        settings.increment_calls("news_agent")
        summary = llm.invoke(summary_prompt)
        state['news'] = summary.content
        state['status'] = "news_complete"

        # 3. Log Success to Database
        db = SessionLocal()
        try:
            log_agent(db, session_id, "news_agent", "completed", state['news'])
        finally:
            db.close()

        print(f"News Agent Complete for {state['company']}")

    except Exception as e:
        error_msg = f"News Agent Failed: {str(e)}"
        state['errors'].append(error_msg)
        state['news'] = "News Data Unavailable."
        state['status'] = "news_agent_failed"
        
        print(f"Error in News Agent: {error_msg}")

        # Save the error state to the database
        db = SessionLocal()
        try:
            log_agent(db, session_id, "news_agent", "failed", error_msg)
        finally:
            db.close()

    return state