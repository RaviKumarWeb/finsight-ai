import json
import time
import requests
import yfinance as yf
from langchain_tavily import TavilySearch
from graph.state import ResearchState
from database.crud import log_agent
from database.connection import SessionLocal
from config.settings import settings

def get_ticker_symbol_direct(company: str) -> str | None:
    try:
        # Clean the name: Remove common suffixes that confuse search APIs
        clean_name = company.replace("Limited", "").replace("Ltd", "").replace("Inc", "").strip()
        
        search_url = "https://query2.finance.yahoo.com/v1/finance/search"
        # A robust User-Agent is critical to avoid being blocked/ignored
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        params = {"q": clean_name, "quotes_count": 5} 
        
        res = requests.get(search_url, params=params, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            quotes = data.get('quotes', [])
            
            if not quotes:
                return None

            # 1. Look for Indian NSE (.NS) or BSE (.BO) matches specifically
            indian_matches = [q for q in quotes if q.get('symbol', '').endswith(('.NS', '.BO'))]
            if indian_matches:
                return indian_matches[0]['symbol']

            # 2. Fallback to the first EQUITY match (Primary listing)
            equities = [q for q in quotes if q.get('quoteType') == 'EQUITY']
            return equities[0]['symbol'] if equities else quotes[0]['symbol']
            
    except Exception as e:
        print(f"Ticker Search Debug: {str(e)}")
        return None


def financial_agent(state: ResearchState, llm) -> ResearchState:
    print(f"Financial Agent Running for: {state['company']}")
    
    # Initialize variables
    state['errors'] = state.get('errors', [])
    session_id = state.get('session_id', 'unknown')
    
    try:
        # STEP 1: Get Ticker (Zero API Cost)
        ticker_symbol = get_ticker_symbol_direct(state["company"])
        
        # STEP 2: Fetch Financial Data (Zero API Cost)
        financial_metrics = {}
        currency = "USD"
        exchange = "Unknown"

        if ticker_symbol:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            currency = info.get("currency", "USD")
            exchange = info.get("exchange", "Global")
            
            financial_metrics = {
                "Ticker": ticker_symbol,
                "Exchange": exchange,
                "Currency": currency,
                "Current Price": info.get("currentPrice"),
                "Market Cap": info.get("marketCap"),
                "Revenue": info.get("totalRevenue"),
                "Profit Margin": info.get("profitMargins"),
                "52 Week High": info.get("fiftyTwoWeekHigh"),
                "52 Week Low": info.get("fiftyTwoWeekLow"),
                "PE Ratio": info.get("trailingPE")
            }
        else:
            print(f"Warning: No ticker found for {state['company']}. Proceeding with News only.")

        # STEP 3: Search for News (Tavily Call)
        search = TavilySearch(max_results=5)
        search_query = f"{state['company']} {ticker_symbol if ticker_symbol else ''} latest financial results 2026 performance"
        search_results = search.invoke(search_query)

        # STEP 4: THE ONLY GROQ CALL
        # We combine Ticker validation and Analysis here
        summary_prompt = f"""
        Analyze the financial health of {state['company']}.
        
        [CONTEXT]
        - Identified Ticker: {ticker_symbol if ticker_symbol else 'Unknown'}
        - Primary Currency: {currency}
        - Financial Data: {json.dumps(financial_metrics, indent=2)}
        - Latest News: {search_results}

        [INSTRUCTIONS]
        1. If the ticker provided is missing or looks wrong for the company's home market, mention the correct one.
        2. Provide a summary of valuation, revenue trends, and 2026 outlook.
        3. Use the appropriate currency symbol (e.g., ₹ for INR, $ for USD).
        """
        settings.increment_calls("financial_agent")
        summary = llm.invoke(summary_prompt)
        state['financial_data'] = summary.content
        state['status'] = "financial_complete"

        # Log Success to DB
        db = SessionLocal()
        try:
            log_agent(db, session_id, "financial_agent", "completed", state['financial_data'])
        finally:
            db.close()

        print(f"Financial Agent Complete for {state['company']} ({currency})")

    except Exception as e:
        error_msg = f"Financial Agent Error: {str(e)}"
        state['errors'].append(error_msg)
        state['financial_data'] = "Financial analysis unavailable."
        state['status'] = "financial_failed"
        
        print(f"Error in Financial Agent: {error_msg}")

        db = SessionLocal()
        try:
            log_agent(db, session_id, "financial_agent", "failed", error_msg)
        finally:
            db.close()

    return state
