DETECT_LANGUAGE = """
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

ROUTE_QUERY = """
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

TRANSLATE_QUERY = """
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

REWRITE_QUERY = """
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

RERANK_PASSAGES = """
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

FINAL_ANSWER = """
You are a spiritual scripture teacher.
Answer the user's question using ONLY the provided context.

Context:
{context_text}

Question (English):
{query}

Answer Language: {language}

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
