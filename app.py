import logging
from typing import Optional as _Optional

import streamlit as st
from openai import OpenAI

from coderag.config import OPENAI_API_KEY, DEEPSEEK_API_KEY, DEEPSEEK_API_BASE
from prompt_flow import execute_rag_flow

# Configure logging for Streamlit
# Use force=True to ensure Streamlit's default handlers don't suppress ours
logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

# Initialize the DeepSeek client with fallback to OpenAI
client: _Optional[OpenAI]
try:
    if DEEPSEEK_API_KEY:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)
        logger.info("DeepSeek client initialized successfully")
    elif OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized successfully")
    else:
        client = None
        logger.error("Neither DeepSeek nor OpenAI API key found")
except Exception as e:
    client = None
    logger.error(f"Failed to initialize AI client: {e}")

# Set page config
st.set_page_config(
    page_title="CodeRAG: Your Coding Assistant", page_icon="🤖", layout="wide"
)

st.title("🤖 CodeRAG: Your Coding Assistant")
st.markdown("*AI-powered code retrieval and assistance using RAG technology*")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = []

# Sidebar with controls
with st.sidebar:
    st.header("Controls")

    if st.button("🗑️ Clear Conversation", type="secondary"):
        st.session_state.messages = []
        st.session_state.conversation_context = []
        st.rerun()

    # Status indicators
    st.header("Status")
    if client:
        if DEEPSEEK_API_KEY:
            st.success("✅ DeepSeek Connected")
        else:
            st.success("✅ OpenAI Connected")
    else:
        st.error("❌ AI Service Not Connected")
        st.error("Please check your API keys in .env file")

    # Conversation stats
    if st.session_state.messages:
        st.info(f"💬 {len(st.session_state.messages)} messages in conversation")

# Display chat history with improved formatting
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "error" in message["content"].lower():
            st.error(message["content"])
        else:
            st.markdown(message["content"])

# Chat input with validation
if not client:
    st.warning(
        "⚠️ OpenAI client not available. Please configure your API key to use "
        "the assistant."
    )
    st.stop()

if prompt := st.chat_input("What is your coding question?", disabled=not client):
    # Validate input
    if not prompt.strip():
        st.warning("Please enter a valid question.")
        st.stop()

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Add to conversation context for better continuity
    st.session_state.conversation_context.append(f"User: {prompt}")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # Show loading indicator
        with st.spinner("🔍 Searching codebase and generating response..."):
            try:
                # Execute RAG flow with error handling
                response = execute_rag_flow(prompt)

                # Check if response indicates an error
                if (
                    response.startswith("Error:")
                    or "error occurred" in response.lower()
                ):
                    message_placeholder.error(response)
                else:
                    message_placeholder.markdown(response)

                full_response = response

            except Exception as e:
                error_message = f"Unexpected error: {str(e)}"
                logger.error(f"Streamlit error: {error_message}")
                message_placeholder.error(error_message)
                full_response = error_message

        # Add assistant response to session
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        # Add to conversation context
        st.session_state.conversation_context.append(
            f"Assistant: {full_response[:200]}..."
        )  # Truncate for context

        # Keep conversation context manageable (last 10 exchanges)
        if len(st.session_state.conversation_context) > 20:
            st.session_state.conversation_context = (
                st.session_state.conversation_context[-20:]
            )

# Footer with helpful information
if not st.session_state.messages:
    st.markdown("---")
    st.markdown("### 💡 Tips for better results:")
    st.markdown(
        """
    - Ask specific questions about your code
    - Mention file names or functions you're interested in
    - Request explanations, improvements, or debugging help
    - Ask about code patterns or best practices
    """
    )

    st.markdown("### 🚀 Example queries:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Explain the indexing process"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": "Explain how the FAISS indexing works in this codebase",
                }
            )
            st.rerun()
    with col2:
        if st.button("🐛 Help debug search issues"):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": (
                        "How can I debug issues with code search not returning "
                        "results?"
                    ),
                }
            )
            st.rerun()
