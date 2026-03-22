from sqlalchemy.orm import Session
from database.models import ResearchSession, AgentLog
from datetime import datetime

# Create new research session
def create_session(db: Session, session_id: str, company: str):
    session = ResearchSession(
        id=session_id,
        company=company,
        status="started"
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

# Update session with results
def update_session(db: Session, session_id: str, **kwargs):
    session = db.query(ResearchSession).filter(
        ResearchSession.id == session_id
    ).first()
    if session:
        for key, value in kwargs.items():
            setattr(session, key, value)
        db.commit()
        db.refresh(session)
    return session

# Log agent action
def log_agent(db: Session, session_id: str, 
              agent_name: str, status: str, 
              output_preview: str = None):
    log = AgentLog(
        session_id=session_id,
        agent_name=agent_name,
        status=status,
        output_preview=output_preview[:200] if output_preview else None
    )
    db.add(log)
    db.commit()

# Get session by id
def get_session(db: Session, session_id: str):
    return db.query(ResearchSession).filter(
        ResearchSession.id == session_id
    ).first()

# Get all sessions for a company
def get_company_history(db: Session, company: str):
    return db.query(ResearchSession).filter(
        ResearchSession.company == company
    ).all()


if __name__ == "__main__":
    from database.connection import SessionLocal, create_tables
    import uuid
    
    create_tables()
    db = SessionLocal()
    
    # Test create session
    session_id = str(uuid.uuid4())
    session = create_session(db, session_id, "Tesla")
    print(f"✅ Session created: {session.id}")
    
    # Test log agent
    log_agent(db, session_id, "news_agent", "completed", 
              "Tesla revenue declined 3%...")
    print("✅ Agent logged")
    
    # Test update session
    update_session(db, session_id, 
                   status="completed",
                   verdict="HOLD")
    print("✅ Session updated")
    
    # Test get session
    result = get_session(db, session_id)
    print(f"✅ Retrieved: {result.company} - {result.status}")
    
    db.close()