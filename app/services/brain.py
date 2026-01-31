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
        self.groq_client = None
        self.collection = None
        
        if not self.groq_key:
            print("❌ Brain Error: GROQ_API_KEY not found.")
        else:
            try:
                self.groq_client = Groq(
                    api_key=self.groq_key, 
                    timeout=30.0
                )
            except Exception as e:
                 print(f"❌ Brain Error (Groq Init): {e}")

        try:
            self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            # Check if collection exists or create/get
            self.collection = self.chroma_client.get_collection(name="vachanamrut_rag", embedding_function=self.ef)
            print("🧠 Brain: Connected to Vector Database.")
        except Exception as e:
            print(f"❌ Brain Error (ChromaDB): {e}")
            self.collection = None

    def _call_llm(self, messages, temperature=0.0, json_mode=True, model="llama-3.3-70b-versatile"):
        if not self.groq_client:
            print("❌ LLM Error: Client not initialized (Missing Key?).")
            return {} if json_mode else ""

        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            
            completion = self.groq_client.chat.completions.create(**kwargs)
            content = completion.choices[0].message.content
            if json_mode:
                return json.loads(content)
            return content
        except Exception as e:
            print(f"❌ LLM Error: {e}")
            return {} if json_mode else ""

    # 1. Detect Language
    def detect_language(self, user_query: str):
        prompt = f"""
        Detect the language of the following user query.

        Supported languages:
        - English → en
        - Hindi → hi
        - Gujarati → gu

        User Query:
        {user_query}

        Output JSON ONLY:
        {{ "language": "en" }}
        
        Deterministic. No creativity.
        """
        result = self._call_llm([{"role": "user", "content": prompt}])
        return result.get("language", "en")

    # 2. Routing (The "Thinking" Part)
    def route_query(self, user_query: str, history_context: str):
        prompt = f"""
        You are a scripture routing assistant.

        Your task is to determine whether the user is referring to a specific
        Vachanamrut discourse.

        Use:
        - The user's ORIGINAL language
        - The conversation history
        - Cultural and scripture-specific phrasing

        Conversation History:
        {history_context}

        User Query:
        {user_query}

        Valid Chapters:
        Gadhada, Sarangpur, Kariyani, Loya, Panchala, Vartal, Amdavad, Jetalpur, Ashlali

        Valid Sections:
        I, II, III, Middle, Last

        Rules:
        - If the user says "this", "it", "that discourse", "આ વચનામૃત", or "यह वचनामृत",
        infer the correct Vachanamrut from history.
        - If no specific discourse is referenced, return empty JSON.

        Output JSON ONLY:
        {{
        "chapter": "Gadhada",
        "section": "I",
        "vachanamrut_no": 16
        }}

        OR:
        {{}}
        """
        return self._call_llm([{"role": "user", "content": prompt}])

    # 3. Translation
    def translate_query(self, user_query: str):
        prompt = f"""
        Translate the following text into English.

        Rules:
        - Preserve spiritual and philosophical meaning
        - Preserve references to Vachanamrut chapters and numbers
        - Do NOT simplify
        - Do NOT add explanations

        Text:
        {user_query}

        Output ONLY the translated English text.
        """
        return self._call_llm([{"role": "user", "content": prompt}], json_mode=False)

    # 4. Query Rewrite
    def rewrite_query(self, translated_query: str, routing_metadata: dict):
        prompt = f"""
        You are a query rewriting assistant.

        Your job is to rewrite the query into a clear, explicit search query
        for retrieving Vachanamrut scripture passages.

        Context:
        - Routing Metadata: {routing_metadata}
        - Original Question (English): {translated_query}

        Rules:
        - Resolve vague phrases like "this", "it", "that teaching"
        - Include chapter/section if known
        - Keep it concise and factual
        - Do NOT answer the question

        Output ONLY the rewritten query text.
        """
        return self._call_llm([{"role": "user", "content": prompt}], json_mode=False)

    # 5. Rerank
    def rerank_passages(self, rewritten_query: str, documents: list):
        # Format passages for the prompt
        numbered = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""
        You are a relevance ranking assistant.

        User Query:
        {rewritten_query}

        Retrieved Passages:
        {numbered}

        Task:
        Rank the passages by how well they answer the user's question.

        Rules:
        - Rank by relevance, not similarity
        - Prefer direct explanations over general mentions

        Output JSON ONLY:
        {{
        "ranked_indices": [0, 2, 1]
        }}
        """
        result = self._call_llm([{"role": "user", "content": prompt}])
        return result.get("ranked_indices", [])

    # 6. Final Answer
    def answer_query(self, rewritten_query: str, context_text: str, user_language: str):
        prompt = f"""
        You are a spiritual scripture teacher.

        Answer the user's question using ONLY the provided context.

        Context:
        {context_text}

        Question (English):
        {rewritten_query}

        Answer Language: {user_language}

        Rules:
        - Be accurate and grounded in scripture
        - Do not invent teachings
        - If the context does not fully answer the question, say so honestly
        - Use respectful, calm, and clear language
        - Match cultural tone:
        • Gujarati → formal spiritual Gujarati
        • Hindi → simple, devotional Hindi
        • English → clear explanatory English

        Now provide the answer.
        """
        return self._call_llm([{"role": "user", "content": prompt}], json_mode=False, temperature=0.1)

    # --- STREAMING ORCHESTRATOR ---
    async def process_user_query_stream(self, user_query: str, chat_history: list, manual_filters: dict = None):
        """
        Yields chunks:
        1. {"type": "thought", "data": "Detecting Language..."}
        2. {"type": "citation", "data": [...]}
        3. {"type": "token", "data": "The"}
        """
        
        yield {"type": "thought", "data": "🧠 Analyzing your question..."}
        
        # 0. Context Prep
        history_txt = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history[-4:]]) if chat_history else ""

        # 1. Detect Language
        lang = self.detect_language(user_query)
        yield {"type": "thought", "data": f"🌍 Detected Language: {lang}"}

        # 2. Route (Resolve References)
        routing_meta = self.route_query(user_query, history_txt)
        if routing_meta:
            yield {"type": "thought", "data": f"🧠 Understanding Context: {routing_meta}"}

        # 3. Translate (if needed)
        translated_query = user_query
        if lang != "en":
            yield {"type": "thought", "data": "🌐 Translating for Search..."}
            translated_query = self.translate_query(user_query)
        
        # 4. Rewrite
        search_query = self.rewrite_query(translated_query, routing_meta)
        yield {"type": "thought", "data": "✏️ Searching Scripture..."}

        # 5. Search (ChromaDB)
        final_where = None
        if manual_filters:
            final_where = manual_filters
        elif routing_meta:
            conditions = []
            if routing_meta.get("chapter"): conditions.append({"chapter": routing_meta["chapter"]})
            if routing_meta.get("section"): conditions.append({"section": routing_meta["section"]})
            if routing_meta.get("vachanamrut_no"): conditions.append({"vachanamrut_no": int(routing_meta["vachanamrut_no"])})
            
            if len(conditions) == 1: final_where = conditions[0]
            elif len(conditions) > 1: final_where = {"$and": conditions}

        if not self.collection:
            yield {"type": "error", "data": "Database not ready."}
            return

        results = self.collection.query(
            query_texts=[search_query],
            n_results=5,
            where=final_where
        )

        if not results['documents'] or not results['documents'][0]:
            yield {"type": "token", "data": "I could not find relevant Vachanamruts."}
            return

        # 6. Rerank
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        yield {"type": "thought", "data": "📊 Ranking Results..."}
        ranked_indices = self.rerank_passages(search_query, documents)
        
        # Reorder docs and metas based on rank
        final_docs = []
        final_metas = []
        indices_to_use = ranked_indices if ranked_indices else range(len(documents))
        
        for i in indices_to_use:
            if i < len(documents):
                final_docs.append(documents[i])
                final_metas.append(metadatas[i])
        
        context_text = "\n\n".join(final_docs)
        
        # Send Citations First
        yield {"type": "citation", "data": [
            {"text": d[:100]+"...", "metadata": m} for d, m in zip(final_docs, final_metas)
        ]}

        # 7. Answer Stream
        yield {"type": "thought", "data": "💡 Generating Answer..."}
        
        prompt = f"""
        You are a spiritual scripture teacher.
        Answer the user's question using ONLY the provided context.

        Context:
        {context_text}

        Question (English):
        {rewritten_query if 'rewritten_query' in locals() else search_query}

        Answer Language: {lang}

        Rules:
        - Be accurate and grounded in scripture
        - Do not invent teachings
        - If the context does not fully answer the question, say so honestly
        - Use respectful, calm, and clear language
        - Match cultural tone:
        • Gujarati → formal spiritual Gujarati
        • Hindi → simple, devotional Hindi
        • English → clear explanatory English

        Now provide the answer.
        """
        
        try:
            stream = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield {"type": "token", "data": chunk.choices[0].delta.content}
                    
        except Exception as e:
            print(f"Streaming Error: {e}")
            yield {"type": "error", "data": str(e)}

# Singleton Instance
brain_service = Brain(db_path="./data/vachanamrut_db")