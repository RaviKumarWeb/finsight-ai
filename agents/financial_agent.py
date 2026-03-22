import json
import time
import yfinance as yf
from langchain_tavily import TavilySearch
from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings


# Add this function above financial_agent
def get_ticker_symbol(company: str, llm) -> str:
    response = llm.invoke(f"""
    Return ONLY the stock ticker symbol for "{company}".
    Exchange suffixes by country:
    - India NSE: .NS (e.g. INFY.NS, TCS.NS)
    - India BSE: .BO
    - UK: .L
    - Germany: .DE  
    - Japan: .T
    - Hong Kong: .HK
    - US stocks: no suffix (e.g. AAPL, TSLA)
    Return ONLY the ticker symbol, nothing else.
    """)
    return response.content.strip()

def financial_agent(state: ResearchState,llm) -> ResearchState:
    print(f"Financial Agent Running for: {state['company']}")

    try:
        ticker_symbol = get_ticker_symbol(state["company"], llm)
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        data={
            "Current Price": info.get("currentPrice"),
            "Market Cap": info.get("marketCap"),
            "Revenue": info.get("totalRevenue"),
            "Profit Margin": info.get("profitMargins"),
            "52 Week High": info.get("fiftyTwoWeekHigh"),
            "52 Week Low": info.get("fiftyTwoWeekLow"),
            "PE Ratio": info.get("trailingPE")
        }

        search = TavilySearch(max_results=5)
        results = search.invoke(f"{state['company']} earnings revenue profit financial results 2026")

        time.sleep(1)

        summary = llm.invoke(f"""
        Summarize key financial metrics for {state['company']}.
        Include specific numbers.
        Cover: stock price, revenue, profit, growth rate, analyst outlook.

        Finance Data: {json.dumps(data, indent=2)}
        News: {results}
        """)

        state['financial_data'] = summary.content
        state['status'] = "financial_complete"

        db = SessionLocal()
        log_agent(db, state['session_id'], "financial_agent", "completed",state['financial_data'])
        db.close()

        print("Financial Agent Complete")


    except Exception as e:
        state['errors'].append(f"Financial Agent Failed: Error {str(e)}")
        state['financial_data'] = "Financial Data Unavailable"
        state['status']= "financial_agent_failed"

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
    
    result = financial_agent(test_state, llm)
    print(result["financial_data"])