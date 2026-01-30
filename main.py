from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# Import our Clean Modules
from app.models.schemas import QueryRequest, AIResponse, Citation
from app.services.librarian import librarian_service
from app.services.brain import brain_service

app = FastAPI(title="Vachanamrut AI API")

origins = [
    "http://localhost:5173",  
    "http://127.0.0.1:5173",    # <--- Add this
    "*"                         # <--- Add this TEMPORARILY to debug               # For local development   # <--- YOUR RENDER FRONTEND URL
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
    return {"status": "ok", "modules": ["Brain", "Librarian"]}

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

# --- UPDATED CHAT ENDPOINT ---
@app.post("/ask", response_model=AIResponse)
def ask_ai(request: QueryRequest):
    
    # A. Check for Manual Filters (Sidebar)
    conditions = []
    if request.chapter != "All": conditions.append({"chapter": request.chapter})
    if request.section != "All": conditions.append({"section": request.section})
    if request.vachanamrut_no > 0: conditions.append({"vachanamrut_no": request.vachanamrut_no})
    
    manual_clause = None
    if len(conditions) == 1: manual_clause = conditions[0]
    elif len(conditions) > 1: manual_clause = {"$and": conditions}

    # B. Ask the Brain (NOW PASSING HISTORY)
    results = brain_service.search(
        query=request.question,
        history=request.history,  # <--- Critical Update
        manual_filters=manual_clause
    )
    
    # C. Handle Results
    if not results['documents'] or not results['documents'][0]:
         return AIResponse(answer="I could not find relevant information in the Vachanamrut for that request.", citations=[])

    docs = results['documents'][0]
    metas = results['metadatas'][0]
    context_text = "\n\n".join(docs)

    # D. Generate
    system_prompt = "You are a humble spiritual assistant. Answer based strictly on the provided context."
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {request.question}"
    
    answer = brain_service.generate_answer(system_prompt, user_prompt)

    citations = [Citation(text=d, metadata=m) for d, m in zip(docs, metas)]
    
    return AIResponse(answer=answer, citations=citations)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)