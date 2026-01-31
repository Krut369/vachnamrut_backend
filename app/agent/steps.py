import json
from app.services.llm_service import llm_service
from app.agent import prompts

async def detect_language(user_query: str):
    prompt = prompts.DETECT_LANGUAGE.format(user_query=user_query)
    response = await llm_service.generate_response(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True
    )
    content = json.loads(response.choices[0].message.content)
    return content.get("language", "en")

async def route_query(user_query: str, history_context: str):
    prompt = prompts.ROUTE_QUERY.format(
        user_query=user_query, 
        history_context=history_context
    )
    response = await llm_service.generate_response(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True
    )
    return json.loads(response.choices[0].message.content)

async def translate_query(user_query: str):
    prompt = prompts.TRANSLATE_QUERY.format(user_query=user_query)
    response = await llm_service.generate_response(
        messages=[{"role": "user", "content": prompt}],
        json_mode=False
    )
    return response.choices[0].message.content

async def rewrite_query(translated_query: str, routing_metadata: dict):
    prompt = prompts.REWRITE_QUERY.format(
        translated_query=translated_query,
        routing_metadata=routing_metadata
    )
    response = await llm_service.generate_response(
        messages=[{"role": "user", "content": prompt}],
        json_mode=False
    )
    return response.choices[0].message.content

async def rerank_passages(rewritten_query: str, documents: list):
    numbered = "\n".join([f"[{i}] {doc}" for i, doc in enumerate(documents)])
    prompt = prompts.RERANK_PASSAGES.format(
        rewritten_query=rewritten_query,
        numbered=numbered
    )
    response = await llm_service.generate_response(
        messages=[{"role": "user", "content": prompt}],
        json_mode=True
    )
    content = json.loads(response.choices[0].message.content)
    return content.get("ranked_indices", [])
