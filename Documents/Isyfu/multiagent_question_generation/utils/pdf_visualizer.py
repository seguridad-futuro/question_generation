"""
PDF Visualizer: Genera PDF nuevo con texto coloreado por chunk.

Este módulo genera un PDF nuevo donde cada chunk aparece con un color de fondo diferente,
manteniendo el texto original y saltos de línea.
"""

from pathlib import Path
from typing import List, Tuple
import logging

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
except ImportError:
    raise ImportError("reportlab is required. Install with: pip install reportlab")

logger = logging.getLogger(__name__)

# Paleta de 10 colores distintos (HEX)
COLOR_PALETTE_10 = [
    "#FFE5B4",  # Peach
    "#B4D7FF",  # Light Blue
    "#C1FFC1",  # Light Green
    "#FFB4E5",  # Light Pink
    "#FFFACD",  # Light Yellow
    "#E5D7FF",  # Light Purple
    "#FFD7B4",  # Light Orange
    "#D7FFFF",  # Light Cyan
    "#FFD7D7",  # Light Red
    "#D7FFD7",  # Mint Green
]


def get_color_for_chunk(chunk_index: int) -> str:
    """Obtiene un color HEX para un chunk."""
    color_index = chunk_index % len(COLOR_PALETTE_10)
    return COLOR_PALETTE_10[color_index]


def add_colored_backgrounds_to_pdf(
    chunks: List,
    original_file_path: Path,
    output_dir: Path = None
) -> Path:
    """Genera un PDF nuevo con cada chunk en un color diferente."""

    if not chunks:
        raise ValueError("No chunks provided")

    original_file_path = Path(original_file_path)

    if output_dir is None:
        output_dir = Path("output/colored_chunks")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{original_file_path.stem}_highlighted.pdf"
    logger.info(f"Creating colored PDF: {original_file_path.name}")

    try:
        # Crear PDF con reportlab
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch,
        )

        elements = []
        styles = getSampleStyleSheet()

        # Título
        title_style = ParagraphStyle(
            'Title',
            parent=styles['Heading1'],
            fontSize=16,
            textColor=colors.black,
            spaceAfter=20,
            alignment=1  # Center
        )
        elements.append(Paragraph(f"Visualización de Chunks: {original_file_path.stem}", title_style))
        elements.append(Spacer(1, 0.3*inch))

        # Procesar cada chunk
        for chunk in chunks:
            chunk_index = chunk.metadata.get("chunk_index", 0)
            chunk_color_hex = get_color_for_chunk(chunk_index)

            # Crear estilo para este chunk con color de fondo
            chunk_style = ParagraphStyle(
                f'Chunk{chunk_index}',
                parent=styles['BodyText'],
                fontSize=10,
                leading=14,
                textColor=colors.black,
                backColor=colors.HexColor(chunk_color_hex),
                leftIndent=10,
                rightIndent=10,
                spaceBefore=10,
                spaceAfter=10,
                borderPadding=10,
            )

            # Limpiar el contenido del chunk
            content = chunk.content

            # Remover marcadores de página si existen
            if "--- Página" in content:
                # Mantener solo el contenido, sin los marcadores
                parts = content.split("--- Página")
                content = parts[0].strip()
                if len(parts) > 1:
                    # Si hay más contenido después del marcador
                    for part in parts[1:]:
                        # Quitar el número de página pero mantener el texto
                        if "---" in part:
                            text_after_marker = part.split("---", 1)[1].strip()
                            if text_after_marker:
                                content += "\n\n" + text_after_marker

            # Escapar caracteres especiales de XML/HTML
            content = content.replace("&", "&amp;")
            content = content.replace("<", "&lt;")
            content = content.replace(">", "&gt;")

            # Mantener saltos de línea convirtiéndolos a <br/>
            content = content.replace("\n", "<br/>")

            # Agregar encabezado del chunk
            header_style = ParagraphStyle(
                'ChunkHeader',
                parent=styles['Heading3'],
                fontSize=9,
                textColor=colors.HexColor("#333333"),
                spaceAfter=5,
            )

            chunk_num = chunk_index + 1
            total = chunk.metadata.get("total_chunks", len(chunks))
            page_info = f" (Página {chunk.page})" if chunk.page else ""

            elements.append(Paragraph(
                f"<b>Chunk {chunk_num}/{total}</b>{page_info}",
                header_style
            ))

            # Agregar contenido del chunk con color de fondo
            elements.append(Paragraph(content, chunk_style))
            elements.append(Spacer(1, 0.2*inch))

        # Construir PDF
        doc.build(elements)

        logger.info(f"✓ Colored PDF created: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error creating colored PDF: {e}")
        raise


def create_colored_pdfs_batch(chunks_by_document: dict, output_dir: Path = None) -> dict:
    """Crea PDFs coloreados para múltiples documentos."""

    if output_dir is None:
        output_dir = Path("output/colored_chunks")

    results = {}
    errors = []

    for original_path, chunks in chunks_by_document.items():
        try:
            output_path = add_colored_backgrounds_to_pdf(chunks, original_path, output_dir)
            results[str(original_path)] = str(output_path)
            logger.info(f"✓ Generated colored PDF for {Path(original_path).name}")
        except Exception as e:
            logger.error(f"Failed to generate colored PDF for {original_path}: {e}")
            errors.append({"file": str(original_path), "error": str(e)})

    if errors:
        logger.warning(f"⚠ {len(errors)} PDF generation(s) failed")

    return results


__all__ = [
    "add_colored_backgrounds_to_pdf",
    "create_colored_pdfs_batch",
    "get_color_for_chunk",
    "COLOR_PALETTE_10",
]

