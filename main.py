from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import json
import asyncio

# Import our Clean Modules
from app.models.schemas import QueryRequest, AIResponse, Citation
from app.services.librarian import librarian_service
# New Agent Orchestrator
from app.agent.orchestrator import process_user_query_stream

app = FastAPI(title="Vachanamrut AI API")

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "modules": ["Agent", "Librarian", "VectorDB"]}

@app.get("/vachanamrut")
def get_vachanamrut(
    chapter: str = Query(..., description="Chapter Name"),
    number: int = Query(..., description="Number"),
    section: str = Query("", description="Section (Optional)")
):
    result = librarian_service.get_full_text(chapter, section, number)
    if not result:
        raise HTTPException(status_code=404, detail="Vachanamrut not found")
    return result

# --- AGENT STREAMING ENDPOINT ---
@app.post("/ask")
async def ask_ai(request: QueryRequest):
    
    # A. Check for Manual Filters (Sidebar)
    conditions = []
    if request.chapter != "All": conditions.append({"chapter": request.chapter})
    if request.section != "All": conditions.append({"section": request.section})
    if request.vachanamrut_no > 0: conditions.append({"vachanamrut_no": request.vachanamrut_no})
    
    manual_clause = None
    if len(conditions) == 1: manual_clause = conditions[0]
    elif len(conditions) > 1: manual_clause = {"$and": conditions}

    # Generator for Streaming
    async def event_generator():
        try:
            # Use the new modular orchestrator
            async for chunk in process_user_query_stream(
                user_query=request.question,
                chat_history=request.history,
                manual_filters=manual_clause
            ):
                # Format: "data: {JSON}\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)