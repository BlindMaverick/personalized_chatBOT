import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from src.constants import EMBEDDING_MODEL_PATH, OLLAMA_MODEL_NAME
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

PACKAGE_IMPORTS: Dict[str, str] = {
    "streamlit": "streamlit",
    "sentence-transformers": "sentence_transformers",
    "pypdf2": "PyPDF2",
    "pytesseract": "pytesseract",
    "pillow": "PIL",
    "opensearch-py": "opensearchpy",
    "torch": "torch",
    "numpy": "numpy",
    "requests": "requests",
    "ollama": "ollama",
    "openpyxl": "openpyxl",
    "python-pptx": "pptx",
}


def read_requirements() -> List[str]:
    """Reads package requirement lines from requirements.txt."""
    requirements_path = Path("requirements.txt")
    if not requirements_path.exists():
        return []

    packages: List[str] = []
    for line in requirements_path.read_text().splitlines():
        requirement = line.strip()
        if requirement and not requirement.startswith("#"):
            packages.append(requirement)
    return packages


def get_missing_packages(requirements: List[str]) -> List[str]:
    """Returns requirement entries whose import targets are not installed."""
    missing_packages: List[str] = []
    for requirement in requirements:
        package_name = requirement.split("==")[0]
        import_name = PACKAGE_IMPORTS.get(package_name, package_name.replace("-", "_"))
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(requirement)
    return missing_packages


def install_packages(packages: List[str]) -> Tuple[bool, str]:
    """Installs missing packages with the current Python interpreter."""
    if not packages:
        return True, "All required Python packages are already installed."

    command = [sys.executable, "-m", "pip", "install", *packages]
    logger.info("Installing missing packages: %s", packages)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("Package installation failed: %s", result.stderr.strip())
        return False, result.stderr.strip() or "pip install failed."

    logger.info("Installed missing packages successfully.")
    return True, "Installed missing Python packages successfully."


def ensure_embedding_model() -> Tuple[bool, str]:
    """Downloads the sentence-transformer model into the local cache if needed."""
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(EMBEDDING_MODEL_PATH)
    except Exception as exc:
        logger.error("Embedding model setup failed: %s", exc)
        return False, f"Embedding model setup failed: {exc}"

    logger.info("Embedding model is ready: %s", EMBEDDING_MODEL_PATH)
    return True, f"Embedding model ready: {EMBEDDING_MODEL_PATH}"


def ensure_ollama_model() -> Tuple[bool, str]:
    """Ensures the configured Ollama model is available locally."""
    try:
        import ollama

        response = ollama.list()
        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])

        model_names = []
        for model in models or []:
            model_name = getattr(model, "model", None)
            if model_name is None and isinstance(model, dict):
                model_name = model.get("model") or model.get("name")
            if model_name:
                model_names.append(model_name)

        if OLLAMA_MODEL_NAME not in model_names:
            logger.info("Pulling Ollama model: %s", OLLAMA_MODEL_NAME)
            ollama.pull(OLLAMA_MODEL_NAME)
    except Exception as exc:
        logger.warning("Ollama model setup skipped: %s", exc)
        return False, (
            f"Ollama model setup could not complete automatically. "
            f"Start Ollama and run `ollama pull {OLLAMA_MODEL_NAME}` if needed."
        )

    logger.info("Ollama model is ready: %s", OLLAMA_MODEL_NAME)
    return True, f"Ollama model ready: {OLLAMA_MODEL_NAME}"


@st.cache_resource(show_spinner=False)
def bootstrap_runtime() -> Dict[str, object]:
    """Installs missing dependencies and preloads local models."""
    requirements = read_requirements()
    missing_packages = get_missing_packages(requirements)
    packages_ok, package_message = install_packages(missing_packages)

    embedding_ok = False
    embedding_message = "Embedding model setup skipped."
    ollama_ok = False
    ollama_message = "Ollama model setup skipped."

    if packages_ok:
        importlib.invalidate_caches()
        embedding_ok, embedding_message = ensure_embedding_model()
        ollama_ok, ollama_message = ensure_ollama_model()

    success = packages_ok and embedding_ok
    return {
        "success": success,
        "packages_ok": packages_ok,
        "embedding_ok": embedding_ok,
        "ollama_ok": ollama_ok,
        "messages": [package_message, embedding_message, ollama_message],
    }
