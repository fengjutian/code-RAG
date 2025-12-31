import logging
from typing import Optional as _Optional

from openai import OpenAI

from coderag.config import (
    OPENAI_API_KEY, OPENAI_CHAT_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_CHAT_MODEL, DEEPSEEK_API_BASE
)
from coderag.search import search_code

logger = logging.getLogger(__name__)

# Initialize DeepSeek client with fallback to OpenAI
client: _Optional[OpenAI]
try:
    if not DEEPSEEK_API_KEY:
        raise ValueError("DeepSeek API key not found")
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
    logger.info(f"DeepSeek client initialized with chat model: {DEEPSEEK_CHAT_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize DeepSeek client: {e}")
    logger.info("Falling back to OpenAI client")
    try:
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found")
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info(f"OpenAI client initialized with chat model: {OPENAI_CHAT_MODEL}")
    except Exception as openai_e:
        logger.error(f"Failed to initialize OpenAI client: {openai_e}")
        client = None

SYSTEM_PROMPT = (
    "You are an expert coding assistant. Your task is to help users with their "
    "question. Use the retrieved code context to inform your responses, but feel "
    "free to suggest better solutions if appropriate."
)

PRE_PROMPT = (
    "Based on the user's query and the following code context, provide a helpful "
    "response. If improvements can be made, suggest them with explanations.\n\n"
    "User Query: {query}\n\n"
    "Retrieved Code Context:\n{code_context}\n\nYour response:"
)


def execute_rag_flow(user_query: str) -> str:
    """Execute the RAG flow for answering user queries.

    Args:
        user_query: The user's question or request

    Returns:
        AI-generated response based on code context
    """
    try:
        if not client:
            logger.error("OpenAI client not initialized")
            return (
                "Error: AI service is not available. Please check your "
                "OpenAI API key."
            )

        if not user_query or not user_query.strip():
            logger.warning("Empty query received")
            return "Please provide a question or request."

        logger.info(f"Processing query: '{user_query[:50]}...'")

        # Perform code search
        search_results = search_code(user_query)

        if not search_results:
            logger.info("No relevant code found for query")
            return (
                "No relevant code found for your query. The codebase might not be "
                "indexed yet or your query might be too specific."
            )

        logger.debug(f"Found {len(search_results)} search results")

        # Prepare code context with error handling
        try:
            code_context = "\n\n".join(
                [
                    (
                        f"File: {result['filename']}\n"
                        f"Path: {result['filepath']}\n"
                        # Cosine similarity (IndexFlatIP returns inner product)
                        f"Similarity: {max(0.0, min(1.0, result['distance'])):.3f}\n"
                        f"{result['content']}"
                    )
                    for result in search_results[:3]  # Limit to top 3 results
                ]
            )
        except (KeyError, TypeError) as e:
            logger.error(f"Error preparing code context: {e}")
            return "Error processing search results. Please try again."

        # Construct the full prompt
        full_prompt = PRE_PROMPT.format(query=user_query, code_context=code_context)

        # Generate response using OpenAI with error handling
        try:
            logger.debug("Sending request to OpenAI")
            # Rough heuristic: keep total under ~7000 tokens
            est_prompt_tokens = max(1, len(full_prompt) // 4)
            max_completion = max(256, min(2000, 7000 - est_prompt_tokens))
            # Use DeepSeek model if available, otherwise fallback to OpenAI
            chat_model = DEEPSEEK_CHAT_MODEL if DEEPSEEK_API_KEY else OPENAI_CHAT_MODEL
            response = client.chat.completions.create(
                model=chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.3,
                max_tokens=max_completion,
                timeout=60,
            )

            if not response.choices or not response.choices[0].message.content:
                logger.error("Empty response from OpenAI")
                return "Error: Received empty response from AI service."

            result = response.choices[0].message.content.strip()
            logger.info("Successfully generated response")
            return result

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return "Error communicating with AI service. Please try again later."

    except Exception as e:
        logger.error(f"Unexpected error in RAG flow: {str(e)}")
        return "An unexpected error occurred. Please try again."
