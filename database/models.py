from sqlalchemy import Column, String, Float, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase
from datetime import datetime
import uuid


class Base(DeclarativeBase):
    pass

class ResearchSession(Base):
    __tablename__ = "research_session"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4))
    company= Column(String, nullable=False)
    status = Column(String, default="started")
    final_report= Column(Text, nullable=True)
    verdict=Column(String, nullable=True)
    confidence_score= Column(Float, nullable=True)
    red_flags = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AgentLog(Base):
    __tablename__ = "agent_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, nullable=False)
    agent_name = Column(String, nullable=False)
    status = Column(String, nullable=False)
    output_preview = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)