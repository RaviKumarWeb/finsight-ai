from langgraph.graph import StateGraph, END
from graph.state import ResearchState
from agents.news_agent import news_agent
from agents.financial_agent import financial_agent
from agents.risk_agent import risk_agent
from agents.critic_agent import critic_agent
from agents.report_agent import report_agent

def should_continue(state: ResearchState):
    """
    This function decides whether to finish or fix the report.
    """
    if state.get("status") == "critic_approved":
        return END
    # If flagged, send it back to the report agent to try again
    return "report"

def build_graph(llm):
    # 1. Initialize the Graph with the State definition
    workflow = StateGraph(ResearchState)

    # 2. Add Nodes (The Workers)
    workflow.add_node("news", lambda state: news_agent(state, llm))
    workflow.add_node("financial", lambda state: financial_agent(state, llm))
    workflow.add_node("risk", lambda state: risk_agent(state, llm))
    workflow.add_node("report", lambda state: report_agent(state, llm))
    workflow.add_node("critic", lambda state: critic_agent(state, llm))

    # 3. Defining the Flow (The Edges)
    workflow.set_entry_point("news")
    
    workflow.add_edge("news", "financial")
    workflow.add_edge("financial", "risk")
    workflow.add_edge("risk", "report")
    workflow.add_edge("report", "critic")
    
    # This checks the Critic's verdict before ending
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            END: END,
            "report": "report"
        }
    )

    return workflow.compile()
