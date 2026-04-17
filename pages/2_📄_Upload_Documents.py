import logging
import os
import time

import streamlit as st

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE
from src.embeddings import generate_embeddings, get_embedding_model
from src.ingestion import (
    bulk_index_documents,
    create_index,
    delete_documents_by_document_name,
)
from src.opensearch import get_opensearch_client
from src.ocr import extract_text_from_excel, extract_text_from_pdf, extract_text_from_ppts
from src.utils import chunk_text, setup_logging

# Initialize logger
setup_logging()  # Set up centralized logging configuration
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": extract_text_from_pdf,
    ".pptx": extract_text_from_ppts,
    ".xlsx": extract_text_from_excel,
}

# Set page config with title, icon, and layout
st.set_page_config(page_title="JARVIS 1.0 - Upload", page_icon="📂")

# Custom CSS to style the page and sidebar
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
            radial-gradient(circle at 10% 18%, rgba(88, 214, 255, 0.12), transparent 24%),
            radial-gradient(circle at 88% 12%, rgba(124, 255, 203, 0.10), transparent 24%),
            linear-gradient(135deg, #07111f, #0d1b2f 55%, #050b16 100%);
    }
    [data-testid="stAppViewContainer"] * { color: var(--text); }
    [data-testid="stMainBlockContainer"] {
        max-width: 1080px;
        padding-top: 2.25rem;
        padding-bottom: 2.5rem;
    }
    .sidebar .sidebar-content { background: linear-gradient(180deg, rgba(5, 14, 28, 0.96), rgba(7, 19, 36, 0.96)); color: white; padding: 20px; border-right: 1px solid rgba(88, 214, 255, 0.14); }
    .sidebar h2, .sidebar h4 { color: white; }
    .block-container { background: linear-gradient(180deg, rgba(8, 19, 36, 0.84), rgba(8, 17, 30, 0.7)); border-radius: 28px; padding: 28px 32px 34px; box-shadow: 0 20px 60px rgba(11, 190, 255, 0.18); border: 1px solid var(--panel-border); backdrop-filter: blur(18px); }
    .footer-text { font-size: 0.95rem; font-weight: 600; color: var(--muted); text-align: center; margin-top: 16px; font-family: "JetBrains Mono", monospace; }
    .stButton button { background: linear-gradient(135deg, #59d3ff, #6dffcf) !important; color: #04111f !important; border-radius: 999px !important; padding: 0.72rem 1.25rem !important; font-size: 0.96rem !important; font-weight: 700 !important; border: none !important; box-shadow: 0 14px 32px rgba(88, 214, 255, 0.24) !important; }
    .stButton button:hover { filter: brightness(1.03); }
    .stButton.delete-button button { background: linear-gradient(135deg, #ff7a7a, #ffb26d) !important; color: #1c0c0c !important; font-size: 14px; }
    .stButton.delete-button button:hover { filter: brightness(1.03); }
    h1, h2, h3, h4 { color: #f4fbff !important; letter-spacing: -0.03em; }
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
    [data-testid="stFileUploader"] * {
        color: #f4fbff !important;
    }
    [data-baseweb="input"] *,
    [data-baseweb="select"] *,
    [data-testid="stTextInput"] *,
    [data-testid="stTextArea"] * {
        color: #f4fbff !important;
        -webkit-text-fill-color: #f4fbff !important;
    }
    [data-testid="stExpander"] *,
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stFileUploaderDropzone"] * {
        color: #f4fbff !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(180deg, rgba(10, 28, 52, 0.92), rgba(7, 21, 40, 0.92)) !important;
        border: 2px dashed rgba(88, 214, 255, 0.42) !important;
        border-radius: 14px !important;
        box-shadow: inset 0 0 0 1px rgba(124, 255, 203, 0.08);
        padding-top: 1.4rem !important;
        padding-bottom: 1.4rem !important;
    }
    [data-testid="stFileUploaderDropzone"] > div {
        background-color: transparent !important;
    }
    button[kind="secondary"],
    [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #59d3ff, #6dffcf) !important;
        color: #04111f !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        box-shadow: 0 12px 28px rgba(88, 214, 255, 0.2) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar header
st.sidebar.markdown(
    "<h2 style='text-align: center;'>JARVIS 1.0</h2>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<h4 style='text-align: center;'>Document Ingestion Console</h4>",
    unsafe_allow_html=True,
)

# Footer
st.sidebar.markdown(
    """
    <div class="footer-text">
        SYSTEM STATUS // ONLINE
    </div>
    """,
    unsafe_allow_html=True,
)


def render_upload_page() -> None:
    """
    Renders the document upload page for users to upload and manage PDFs.
    Shows only the documents that are present in the OpenSearch index.
    """

    st.title("JARVIS 1.0 // Upload")
    # Placeholder for the loading spinner at the top
    model_loading_placeholder = st.empty()

    # Display the loading spinner at the top for loading the embedding model
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:
            with st.spinner("Loading models for document processing..."):
                get_embedding_model()
                st.session_state["embedding_models_loaded"] = True
        logger.info("Embedding models loaded.")
        model_loading_placeholder.empty()  # Clear the placeholder after loading

    UPLOAD_DIR = "uploaded_files"
    os.makedirs(UPLOAD_DIR, exist_ok=True)

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

    # Initialize or clear the documents list in session state
    st.session_state["documents"] = []

    # Query OpenSearch to get the list of unique document names
    query = {
        "size": 0,
        "aggs": {"unique_docs": {"terms": {"field": "document_name", "size": 10000}}},
    }
    response = client.search(index=index_name, body=query)
    buckets = response["aggregations"]["unique_docs"]["buckets"]
    document_names = [bucket["key"] for bucket in buckets]
    logger.info("Retrieved document names from OpenSearch.")

    # Load document information from the index
    for document_name in document_names:
        file_path = os.path.join(UPLOAD_DIR, document_name)
        if os.path.exists(file_path):
            text = extract_text_for_file(file_path)
            st.session_state["documents"].append(
                {"filename": document_name, "content": text, "file_path": file_path}
            )
        else:
            st.session_state["documents"].append(
                {"filename": document_name, "content": "", "file_path": None}
            )
            logger.warning(f"File '{document_name}' does not exist locally.")

    if "deleted_file" in st.session_state:
        st.success(
            f"The file '{st.session_state['deleted_file']}' was successfully deleted."
        )
        del st.session_state["deleted_file"]

    # Allow users to select any file type in the picker.
    uploaded_files = st.file_uploader("Upload documents", accept_multiple_files=True)

    if uploaded_files:
        with st.spinner("Uploading and processing documents. Please wait..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name in document_names:
                    st.warning(
                        f"The file '{uploaded_file.name}' already exists in the index."
                    )
                    continue

                file_path = save_uploaded_file(uploaded_file)
                text = extract_text_for_file(file_path)
                if not text.strip():
                    st.warning(
                        f"Skipping '{uploaded_file.name}' because its file type is not supported yet."
                    )
                    os.remove(file_path)
                    continue
                chunks = chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=100)
                embeddings = generate_embeddings(chunks)

                documents_to_index = [
                    {
                        "doc_id": f"{uploaded_file.name}_{i}",
                        "text": chunk,
                        "embedding": embedding,
                        "document_name": uploaded_file.name,
                    }
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
                ]
                bulk_index_documents(documents_to_index)
                st.session_state["documents"].append(
                    {
                        "filename": uploaded_file.name,
                        "content": text,
                        "file_path": file_path,
                    }
                )
                document_names.append(uploaded_file.name)
                logger.info(f"File '{uploaded_file.name}' uploaded and indexed.")

        st.success("Files uploaded and indexed successfully!")

    if st.session_state["documents"]:
        st.markdown("### Uploaded Documents")
        with st.expander("Manage Uploaded Documents", expanded=True):
            for idx, doc in enumerate(st.session_state["documents"], 1):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(
                        f"{idx}. {doc['filename']} - {len(doc['content'])} characters extracted"
                    )
                with col2:
                    delete_button = st.button(
                        "Delete",
                        key=f"delete_{doc['filename']}_{idx}",
                        help=f"Delete {doc['filename']}",
                    )
                    if delete_button:
                        if doc["file_path"] and os.path.exists(doc["file_path"]):
                            try:
                                os.remove(doc["file_path"])
                                logger.info(
                                    f"Deleted file '{doc['filename']}' from filesystem."
                                )
                            except FileNotFoundError:
                                st.error(
                                    f"File '{doc['filename']}' not found in filesystem."
                                )
                                logger.error(
                                    f"File '{doc['filename']}' not found during deletion."
                                )
                        delete_documents_by_document_name(doc["filename"])
                        st.session_state["documents"].pop(idx - 1)
                        st.session_state["deleted_file"] = doc["filename"]
                        time.sleep(0.5)
                        st.rerun()


def save_uploaded_file(uploaded_file) -> str:  # type: ignore
    """
    Saves an uploaded file to the local file system.

    Args:
        uploaded_file: The uploaded file to save.

    Returns:
        str: The file path where the uploaded file is saved.
    """
    UPLOAD_DIR = "uploaded_files"
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    logger.info(f"File '{uploaded_file.name}' saved to '{file_path}'.")
    return file_path


def extract_text_for_file(file_path: str) -> str:
    """
    Extracts text from a supported file type based on its extension.

    Args:
        file_path (str): Path to the saved file.

    Returns:
        str: Extracted text, or an empty string for unsupported file types.
    """
    extension = os.path.splitext(file_path)[1].lower()
    extractor = SUPPORTED_EXTENSIONS.get(extension)
    if not extractor:
        logger.warning(f"Unsupported file type for extraction: {file_path}")
        return ""

    return extractor(file_path)


if __name__ == "__main__":
    if "documents" not in st.session_state:
        st.session_state["documents"] = []
    render_upload_page()
