import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
import json
from dotenv import load_dotenv

load_dotenv()

class Brain:
    def __init__(self, db_path: str):
        self.groq_key = os.getenv("GROQ_API_KEY")
        if not self.groq_key:
            print("❌ Brain Error: GROQ_API_KEY not found.")
        
        self.groq_client = Groq(
            api_key=self.groq_key, 
            timeout=30.0
        )
        
        try:
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            self.collection = self.chroma_client.get_collection(name="vachanamrut_rag", embedding_function=self.ef)
            print("🧠 Brain: Connected to Vector Database.")
        except Exception as e:
            print(f"❌ Brain Error: {e}")

    # --- NEW: Router with Memory ---
    def extract_metadata_from_query(self, user_query: str, chat_history: list):
        """
        Uses LLM + History to detect specific Vachanamruts (handles 'this', 'it').
        """
        # 1. Format History for context
        history_context = ""
        # Get last 4 messages
        recent_msgs = chat_history[-4:] if chat_history else []
        
        for msg in recent_msgs:
            # Handle Pydantic objects (msg.role) or dicts (msg['role'])
            role = getattr(msg, "role", None) or msg.get("role")
            content = getattr(msg, "content", None) or msg.get("content")
            history_context += f"{role}: {content}\n"

        system_prompt = f"""
        You are a Query Router. Analyze the User's latest input.
        
        CONTEXT FROM HISTORY:
        {history_context}
        
        INSTRUCTIONS:
        1. If the user mentions a specific Vachanamrut (e.g. 'Gadhada I 16'), extract it.
        2. If the user says "this", "it", or "that vachanamrut", USE THE HISTORY to identify which Vachanamrut they are referring to.
        
        Valid Chapters: Gadhada, Sarangpur, Kariyani, Loya, Panchala, Vartal, Amdavad, Jetalpur, Ashlali.
        Valid Sections: I, II, III, Middle, Last.
        
        Output JSON ONLY. Format: {{"chapter": "Gadhada", "section": "I", "vachanamrut_no": 16}}
        If no specific Vachanamrut is active or mentioned, return: {{}}
        """
        
        try:
            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0, 
                response_format={"type": "json_object"}
            )
            result = json.loads(completion.choices[0].message.content)
            if result.get("vachanamrut_no"):
                print(f"🧠 Brain Router: Detected {result}")
                return result
            return None
        except Exception as e:
            print(f"Router Error: {e}")
            return None

    # --- UPDATED SEARCH LOGIC ---
    def search(self, query: str, history: list, manual_filters: dict = None):
        
        # 1. Start with Manual Filters (from Sidebar)
        final_filters = manual_filters if manual_filters else {}

        # 2. If no Manual Filters, try Auto-Routing
        if not final_filters:
            auto_filters = self.extract_metadata_from_query(query, history)
            if auto_filters:
                conditions = []
                if auto_filters.get("chapter"): conditions.append({"chapter": auto_filters["chapter"]})
                if "section" in auto_filters: conditions.append({"section": auto_filters["section"]})
                if auto_filters.get("vachanamrut_no"): conditions.append({"vachanamrut_no": int(auto_filters["vachanamrut_no"])})
                
                if len(conditions) == 1: final_filters = conditions[0]
                elif len(conditions) > 1: final_filters = {"$and": conditions}

        print(f"🔎 Brain Search Filters: {final_filters}")

        return self.collection.query(
            query_texts=[query],
            n_results=5,
            where=final_filters if final_filters else None
        )

    def generate_answer(self, system_prompt: str, user_prompt: str):
        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        return completion.choices[0].message.content

# Singleton Instance
brain_service = Brain(db_path="./data/vachanamrut_db")