from langchain_tavily import TavilySearch
from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings


def news_agent(state: ResearchState, llm) -> ResearchState:
    print(f" News Agent Running for: {state['company']}")

    try:
        search = TavilySearch(max_results=5)

        results = search.invoke(f"{state['company']} latest news 2026 financial performance")

        summary  = llm.invoke(f"""
        Summarize these news about {state['company']} 
        in 4-5 key bullet points.
        Focus on financial impact and business performance.

        News: {results}
        """)

        state['news'] = summary.content
        state['status'] = "news_complete"

        db = SessionLocal()
        log_agent(db, state['session_id'], "news_agent", "completed",state['news'])
        db.close()

        print("News Agent Complete")
        
    except Exception as e:
        state['errors'].append(f"News Agent Failed: Error {str(e)}")
        state['news'] = "News Data Unavaliable."
        state['status'] = "news_agent_failed"
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
        "company": "Infosis",
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
    
    result = news_agent(test_state, llm)
    print(result['news'])