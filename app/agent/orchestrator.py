from app.agent import steps, prompts
from app.services.vector_service import vector_service
from app.services.llm_service import llm_service

async def process_user_query_stream(user_query: str, chat_history: list, manual_filters: dict = None):
    # 0. Context Prep
    history_txt = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history[-4:]]) if chat_history else ""
    
    yield {"type": "thought", "data": "🧠 Analyzing your question..."}

    # 1. Detect Language
    lang = await steps.detect_language(user_query)
    yield {"type": "thought", "data": f"🌍 Detected Language: {lang}"}

    # 2. Route
    routing_meta = await steps.route_query(user_query, history_txt)
    if routing_meta:
        yield {"type": "thought", "data": f"🧠 Understanding Context: {routing_meta}"}

    # 3. Translate
    translated_query = user_query
    if lang != "en":
        yield {"type": "thought", "data": "🌐 Translating for Search..."}
        translated_query = await steps.translate_query(user_query)

    # 4. Rewrite
    search_query = await steps.rewrite_query(translated_query, routing_meta)
    yield {"type": "thought", "data": "✏️ Searching Scripture..."}

    # 5. Search
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

    if not vector_service.collection:
        yield {"type": "error", "data": "Database not ready."}
        return

    results = vector_service.search(search_query, filters=final_where)

    if not results or not results['documents'] or not results['documents'][0]:
        yield {"type": "token", "data": "I could not find relevant Vachanamruts."}
        return

    # 6. Rerank
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    
    yield {"type": "thought", "data": "📊 Ranking Results..."}
    ranked_indices = await steps.rerank_passages(search_query, documents)

    final_docs = []
    final_metas = []
    indices_to_use = ranked_indices if ranked_indices else range(len(documents))
    
    for i in indices_to_use:
        if i < len(documents):
            final_docs.append(documents[i])
            final_metas.append(metadatas[i])
    
    context_text = "\n\n".join(final_docs)

    # Yield Citations
    yield {"type": "citation", "data": [
        {"text": d[:100]+"...", "metadata": m} for d, m in zip(final_docs, final_metas)
    ]}

    # 7. Answer Stream
    yield {"type": "thought", "data": "💡 Generating Answer..."}

    prompt = prompts.FINAL_ANSWER.format(
        context_text=context_text,
        query=search_query,
        language=lang
    )

    try:
        stream = await llm_service.generate_response(
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield {"type": "token", "data": chunk.choices[0].delta.content}
    except Exception as e:
        yield {"type": "error", "data": str(e)}
