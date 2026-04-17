import logging

import streamlit as st

from src.chat import (  # type: ignore
    ensure_model_pulled,
    generate_response_streaming,
    get_embedding_model,
)
from src.ingestion import create_index, get_opensearch_client
from src.constants import OLLAMA_MODEL_NAME, OPENSEARCH_INDEX
from src.utils import setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(page_title="JARVIS 1.0 - Chat", page_icon="🤖")

# Apply custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');
    :root {
        --bg-1: #07111f;
        --bg-2: #0d1b2f;
        --panel: rgba(8, 19, 36, 0.78);
        --panel-border: rgba(115, 203, 255, 0.22);
        --text: #e7f6ff;
        --muted: #9eb6ca;
        --accent: #58d6ff;
        --accent-2: #7cffcb;
    }
    html, body, [class*="css"] { font-family: "Space Grotesk", sans-serif; }
    body { background: linear-gradient(135deg, #07111f, #0d1b2f 55%, #050b16 100%); color: var(--text); }
    .stApp {
        background:
            radial-gradient(circle at 15% 15%, rgba(88, 214, 255, 0.12), transparent 24%),
            radial-gradient(circle at 85% 10%, rgba(124, 255, 203, 0.10), transparent 24%),
            linear-gradient(135deg, #07111f, #0d1b2f 55%, #050b16 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(5, 14, 28, 0.96), rgba(7, 19, 36, 0.96));
        border-right: 1px solid rgba(88, 214, 255, 0.14);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    .block-container {
        background: linear-gradient(180deg, rgba(8, 19, 36, 0.84), rgba(8, 17, 30, 0.7));
        border: 1px solid var(--panel-border);
        border-radius: 28px;
        padding: 28px 32px 110px;
        box-shadow: 0 20px 60px rgba(11, 190, 255, 0.18);
        backdrop-filter: blur(18px);
    }
    .footer-text { font-size: 0.95rem; font-weight: 600; color: var(--muted); text-align: center; margin-top: 16px; font-family: "JetBrains Mono", monospace; }
    .stButton button {
        background: linear-gradient(135deg, #59d3ff, #6dffcf) !important;
        color: #04111f !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.72rem 1.25rem !important;
        font-size: 0.96rem !important;
        font-weight: 700 !important;
        box-shadow: 0 14px 32px rgba(88, 214, 255, 0.24) !important;
    }
    h1, h2, h3, h4 { color: #f4fbff !important; letter-spacing: -0.03em; }
    p, label, div, span { color: var(--text) !important; }
    .stChatMessage {
        background: linear-gradient(180deg, rgba(11, 32, 59, 0.92), rgba(9, 24, 42, 0.92));
        color: #f4fbff;
        padding: 14px 16px;
        border-radius: 20px;
        margin-bottom: 14px;
        border: 1px solid rgba(88, 214, 255, 0.14);
    }
    .stChatMessage.user {
        background: linear-gradient(135deg, rgba(88, 214, 255, 0.18), rgba(124, 255, 203, 0.18));
        border: 1px solid rgba(124, 255, 203, 0.24);
    }
    [data-testid="stChatMessage"] * { color: #f4fbff !important; }
    [data-testid="stChatInput"] {
        background: rgba(7, 17, 31, 0.92) !important;
        border-radius: 22px;
        border: 1px solid rgba(88, 214, 255, 0.18);
        box-shadow: 0 12px 32px rgba(3, 10, 18, 0.35);
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea::placeholder,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] input::placeholder {
        color: #f4fbff !important;
        -webkit-text-fill-color: #f4fbff !important;
    }
    [data-testid="stChatInput"] textarea {
        background: transparent !important;
        caret-color: #7cffcb !important;
        font-weight: 500;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
        -webkit-text-fill-color: var(--muted) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
logger.info("Custom CSS applied.")


# Main chatbot page rendering function
def render_chatbot_page() -> None:
    # Set up a placeholder at the very top of the main content area
    st.title("JARVIS 1.0 // Chat")
    model_loading_placeholder = st.empty()

    # Initialize session state variables for chatbot settings
    if "use_hybrid_search" not in st.session_state:
        st.session_state["use_hybrid_search"] = True
    if "num_results" not in st.session_state:
        st.session_state["num_results"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7

    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        client = get_opensearch_client()
    index_name = OPENSEARCH_INDEX

    # Ensure the index exists
    try:
        create_index(client)
    except ValueError as exc:
        st.error(str(exc))
        st.info(
            "Current config: model 'sentence-transformers/all-mpnet-base-v2' "
            "expects 768 dimensions."
        )
        st.stop()

    # Sidebar settings for hybrid search toggle, result count, and temperature
    st.session_state["use_hybrid_search"] = st.sidebar.checkbox(
        "Enable RAG mode", value=st.session_state["use_hybrid_search"]
    )
    st.session_state["num_results"] = st.sidebar.number_input(
        "Number of Results in Context Window",
        min_value=1,
        max_value=10,
        value=st.session_state["num_results"],
        step=1,
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )

    # Sidebar headers and footer
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>JARVIS 1.0</h2>", unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Conversational Command Mode</h4>",
        unsafe_allow_html=True,
    )

    # Footer text
    st.sidebar.markdown(
        """
        <div class="footer-text">
            SYSTEM STATUS // ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Sidebar configured with headers and footer.")

    # Display loading spinner at the top of the main content area
    with model_loading_placeholder.container():
        st.spinner("Loading models for chat...")

    # Load models if not already loaded
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:
            with st.spinner("Loading Embedding and Ollama models for Hybrid Search..."):
                get_embedding_model()
                ensure_model_pulled(OLLAMA_MODEL_NAME)
                st.session_state["embedding_models_loaded"] = True
        logger.info("Embedding model loaded.")
        model_loading_placeholder.empty()

    # Initialize chat history in session state if not already present
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input and generate response
    if prompt := st.chat_input("Type your message here..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        logger.info("User input received.")

        # Generate response from assistant
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response_placeholder = st.empty()
                response_text = ""

                response_stream = generate_response_streaming(
                    prompt,
                    use_hybrid_search=st.session_state["use_hybrid_search"],
                    num_results=st.session_state["num_results"],
                    temperature=st.session_state["temperature"],
                    chat_history=st.session_state["chat_history"],
                )

            # Stream response content if response_stream is valid
            if response_stream is not None:
                for chunk in response_stream:
                    if (
                        isinstance(chunk, dict)
                        and "message" in chunk
                        and "content" in chunk["message"]
                    ):
                        response_text += chunk["message"]["content"]
                        response_placeholder.markdown(response_text + "▌")
                    else:
                        logger.error("Unexpected chunk format in response stream.")

            response_placeholder.markdown(response_text)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": response_text}
            )
            logger.info("Response generated and displayed.")


# Main execution
if __name__ == "__main__":
    render_chatbot_page()
