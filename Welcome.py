import logging

import streamlit as st

from src.utils import setup_logging

# Initialize logger
setup_logging()  # Set up logging configuration
logger = logging.getLogger(__name__)

# Set page config with title, icon, and layout
st.set_page_config(
    page_title="JARVIS 1.0", page_icon="🤖"
)


# Custom CSS to style the page and sidebar
def apply_custom_css() -> None:
    """Applies custom CSS styling to the Streamlit page and sidebar."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@500&display=swap');
        :root {
            --bg-1: #07111f;
            --bg-2: #0d1b2f;
            --panel: rgba(10, 20, 38, 0.72);
            --panel-border: rgba(115, 203, 255, 0.22);
            --text: #e7f6ff;
            --muted: #9eb6ca;
            --accent: #58d6ff;
            --accent-2: #7cffcb;
            --glow: 0 20px 60px rgba(11, 190, 255, 0.18);
        }
        html, body, [class*="css"]  {
            font-family: "Space Grotesk", sans-serif;
        }
        body {
            background:
                radial-gradient(circle at top left, rgba(88, 214, 255, 0.18), transparent 30%),
                radial-gradient(circle at top right, rgba(124, 255, 203, 0.14), transparent 24%),
                linear-gradient(135deg, var(--bg-1), var(--bg-2) 55%, #050b16 100%);
            color: var(--text);
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 20%, rgba(88, 214, 255, 0.12), transparent 0, transparent 26%),
                radial-gradient(circle at 85% 10%, rgba(124, 255, 203, 0.10), transparent 0, transparent 24%),
                linear-gradient(135deg, #07111f, #0d1b2f 55%, #050b16 100%);
        }
        [data-testid="stAppViewContainer"] * {
            color: var(--text);
        }
        [data-testid="stMainBlockContainer"] {
            max-width: 1080px;
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }
        .block-container {
            background: linear-gradient(180deg, rgba(8, 19, 36, 0.84), rgba(8, 17, 30, 0.7));
            border: 1px solid var(--panel-border);
            border-radius: 28px;
            padding: 28px 32px 34px;
            box-shadow: var(--glow);
            backdrop-filter: blur(18px);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(5, 14, 28, 0.96), rgba(7, 19, 36, 0.96));
            border-right: 1px solid rgba(88, 214, 255, 0.14);
        }
        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }
        h1, h2, h3, h4 {
            color: #f4fbff !important;
            letter-spacing: -0.03em;
        }
        h1 {
            font-size: 3.1rem !important;
            line-height: 1.02;
            margin-bottom: 0.65rem;
        }
        p, li, label, div {
            color: var(--text) !important;
        }
        strong {
            color: var(--accent-2) !important;
        }
        .footer-text {
            font-size: 0.95rem;
            font-weight: 600;
            color: var(--muted) !important;
            text-align: center;
            margin-top: 16px;
            font-family: "JetBrains Mono", monospace;
        }
        .stButton button,
        button[kind="secondary"],
        [data-testid="stBaseButton-secondary"] {
            background: linear-gradient(135deg, #59d3ff, #6dffcf) !important;
            color: #04111f !important;
            border: none !important;
            border-radius: 999px !important;
            padding: 0.72rem 1.25rem !important;
            font-size: 0.96rem !important;
            font-weight: 700 !important;
            box-shadow: 0 14px 32px rgba(88, 214, 255, 0.24) !important;
            transition: transform 0.18s ease, box-shadow 0.18s ease !important;
        }
        .stButton button:hover,
        button[kind="secondary"]:hover,
        [data-testid="stBaseButton-secondary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 36px rgba(88, 214, 255, 0.28) !important;
        }
        input, textarea, select {
            color: #f4fbff !important;
            -webkit-text-fill-color: #f4fbff !important;
            background: rgba(7, 17, 31, 0.85) !important;
            caret-color: #7cffcb !important;
            border-radius: 16px !important;
        }
        input::placeholder, textarea::placeholder {
            color: var(--muted) !important;
            -webkit-text-fill-color: var(--muted) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Applied custom CSS styling.")


# Function to display main content
def display_main_content() -> None:
    """Displays the main welcome content on the page."""
    st.markdown(
        """
        <div style="display:inline-block;padding:0.35rem 0.8rem;border:1px solid rgba(88,214,255,0.28);border-radius:999px;background:rgba(88,214,255,0.08);font-family:'JetBrains Mono',monospace;font-size:0.82rem;letter-spacing:0.08em;text-transform:uppercase;">
        JARVIS 1.0
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("JARVIS 1.0")
    st.markdown(
        """
        A local AI command console for retrieval, document indexing, and grounded conversation.

        **What you can do here**
        - **Chat with context** using your local LLM setup and retrieved document chunks.
        - **Upload and index documents** into OpenSearch for fast semantic + keyword retrieval.
        - **Stay private** by keeping your workflow on your own machine.

        **Choose a page from the sidebar to launch the workflow.**
        """
    )
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="padding:1.1rem;border:1px solid rgba(88,214,255,0.18);border-radius:22px;background:rgba(8,19,36,0.72);">
            <div style="font-family:'JetBrains Mono',monospace;color:#7cffcb;font-size:0.8rem;">MODULE_01</div>
            <div style="font-size:1.15rem;font-weight:700;margin-top:0.4rem;">Hybrid Search</div>
            <div style="color:#9eb6ca;margin-top:0.35rem;">Blend vector retrieval with exact text matches for grounded answers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div style="padding:1.1rem;border:1px solid rgba(88,214,255,0.18);border-radius:22px;background:rgba(8,19,36,0.72);">
            <div style="font-family:'JetBrains Mono',monospace;color:#7cffcb;font-size:0.8rem;">MODULE_02</div>
            <div style="font-size:1.15rem;font-weight:700;margin-top:0.4rem;">Local Inference</div>
            <div style="color:#9eb6ca;margin-top:0.35rem;">Run the assistant against your own stack without shipping files away.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div style="padding:1.1rem;border:1px solid rgba(88,214,255,0.18);border-radius:22px;background:rgba(8,19,36,0.72);">
            <div style="font-family:'JetBrains Mono',monospace;color:#7cffcb;font-size:0.8rem;">MODULE_03</div>
            <div style="font-size:1.15rem;font-weight:700;margin-top:0.4rem;">Document Ops</div>
            <div style="color:#9eb6ca;margin-top:0.35rem;">Upload, review, and manage the files feeding your knowledge base.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    logger.info("Displayed main welcome content.")


# Function to display sidebar content
def display_sidebar_content() -> None:
    """Displays headers and footer content in the sidebar."""
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>JARVIS 1.0</h2>", unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Autonomous Document Console</h4>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <div class="footer-text">
            SYSTEM STATUS // ONLINE
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Displayed sidebar content.")


# Main execution
if __name__ == "__main__":
    apply_custom_css()
    display_sidebar_content()
    display_main_content()
