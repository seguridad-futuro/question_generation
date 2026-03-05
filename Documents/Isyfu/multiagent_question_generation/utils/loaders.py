"""Document loaders for PDF, TXT, and MD files."""

from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_document(file_path: Path) -> Dict[str, any]:
    """Carga un documento y retorna su contenido con metadata.

    Args:
        file_path: Path al documento

    Returns:
        Dict con {content: str, metadata: dict}

    Raises:
        ValueError: Si el formato no es soportado
        FileNotFoundError: Si el archivo no existe
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".txt":
        return _load_txt(file_path)
    elif suffix == ".md":
        return _load_md(file_path)
    elif suffix == ".pdf":
        return _load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_txt(file_path: Path) -> Dict[str, any]:
    """Carga archivo TXT."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "metadata": {
                "source": str(file_path),
                "type": "text",
                "size": len(content),
            }
        }
    except Exception as e:
        logger.error(f"Error loading TXT {file_path}: {e}")
        raise


def _load_md(file_path: Path) -> Dict[str, any]:
    """Carga archivo Markdown (similar a TXT)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return {
            "content": content,
            "metadata": {
                "source": str(file_path),
                "type": "markdown",
                "size": len(content),
            }
        }
    except Exception as e:
        logger.error(f"Error loading MD {file_path}: {e}")
        raise


def _load_pdf(file_path: Path) -> Dict[str, any]:
    """Carga archivo PDF usando pypdf.

    Args:
        file_path: Path al PDF

    Returns:
        Dict con contenido y metadata (incluyendo número de páginas)
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        num_pages = len(reader.pages)

        # Extraer texto de todas las páginas
        content = ""
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text()
            # Añadir marcador de página para trazabilidad
            content += f"\n\n--- Página {page_num} ---\n\n{page_text}"

        return {
            "content": content,
            "metadata": {
                "source": str(file_path),
                "type": "pdf",
                "num_pages": num_pages,
                "size": len(content),
            }
        }
    except ImportError:
        logger.error("pypdf not installed. Install with: pip install pypdf")
        raise
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        raise


def load_documents(input_dir: Path) -> List[Dict[str, any]]:
    """Carga todos los documentos de un directorio.

    Args:
        input_dir: Directorio con documentos

    Returns:
        Lista de documentos cargados con {content, metadata}
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    documents = []
    supported_extensions = [".pdf", ".txt", ".md"]

    for file_path in input_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                doc = load_document(file_path)
                documents.append(doc)
                logger.info(f"Loaded document: {file_path.name} ({doc['metadata']['type']})")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                continue

    if not documents:
        raise ValueError(f"No valid documents found in {input_dir}")

    logger.info(f"Loaded {len(documents)} documents total")
    return documents
