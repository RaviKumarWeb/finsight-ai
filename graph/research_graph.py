from graph.state import ResearchState
from config.settings import settings
from langgraph.graph import StateGraph, END
from agents.news_agent import news_agent
from agents.financial_agent import financial_agent
from agents.risk_agent import risk_agent
from agents.critic_agent import critic_agent
from agents.report_agent import report_agent
from database.crud import create_session
from database.connection import SessionLocal






def build_graph(llm):

    graph = StateGraph(ResearchState)

    graph.add_node("news", lambda state: news_agent(state,llm))
    graph.add_node("financial", lambda state: financial_agent(state,llm))
    graph.add_node("risk", lambda state: risk_agent(state,llm))
    graph.add_node("critic", lambda state: critic_agent(state,llm))

    graph.add_node("report", lambda state: report_agent(state,llm))


    graph.set_entry_point("news")
    graph.add_edge("news", "financial")
    graph.add_edge("financial", "risk")
    graph.add_edge("risk", "report")
    graph.add_edge("report", "critic")
    graph.add_edge("critic", END)

    return graph.compile()