from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# 1. Define a single message object
class ChatMessage(BaseModel):
    role: str
    content: str

# 2. Update Request to include History
class QueryRequest(BaseModel):
    question: str
    history: List[ChatMessage] = [] # New Field!
    
    # Manual overrides (Sidebar)
    chapter: Optional[str] = "All"
    section: Optional[str] = "All"
    vachanamrut_no: Optional[int] = 0

class Citation(BaseModel):
    text: str
    metadata: Dict[str, Any]

class AIResponse(BaseModel):
    answer: str
    citations: List[Citation]