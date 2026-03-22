from typing import TypedDict, List, Optional

class ResearchState(TypedDict):
    company: str
    news: Optional[str]
    financial_data: Optional[str]
    risk_assessment: Optional[str]
    critic_data: Optional[str]
    final_report: Optional[str]
    document_analysis: Optional[str]
    session_id: Optional[str] 
    confidence_score: Optional[float]
    errors: List[str] 
    status: str

