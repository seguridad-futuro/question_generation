"""Filtro de metadatos de PDF para detectar y eliminar contenido no sustantivo.

Este módulo identifica y filtra metadatos típicos de PDFs como:
- Títulos de documentos/libros
- Autores, editores, fechas de publicación
- Emails, teléfonos, URLs
- ISBN, editorial
- Números de página
- Índices o tablas de contenido
- Headers y footers repetitivos
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class MetadataFilterResult:
    """Resultado del filtrado de metadatos."""

    original_content: str
    clean_content: str
    is_substantive: bool
    substantive_ratio: float
    detected_metadata: List[Dict[str, str]] = field(default_factory=list)
    filter_reasons: List[str] = field(default_factory=list)


class PDFMetadataFilter:
    """Filtro de metadatos para chunks de documentos PDF.

    Detecta y elimina contenido no sustantivo como metadatos,
    información de publicación, índices, etc.
    """

    # ==========================================
    # PATRONES REGEX PARA DETECCIÓN
    # ==========================================

    # Emails
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    )

    # URLs
    URL_PATTERN = re.compile(
        r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+',
        re.IGNORECASE
    )

    # Teléfonos (formato español e internacional)
    PHONE_PATTERN = re.compile(
        r'(?:\+34\s?)?\d{3}[\s.-]?\d{3}[\s.-]?\d{3}|\d{2}[\s.-]?\d{3}[\s.-]?\d{2}[\s.-]?\d{2}',
        re.IGNORECASE
    )

    # ISBN
    ISBN_PATTERN = re.compile(
        r'ISBN[:\s]*(?:97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?\d{1,7}[-\s]?[\dXx]',
        re.IGNORECASE
    )

    # Depósito Legal
    DEPOSITO_LEGAL_PATTERN = re.compile(
        r'(?:Depósito\s*Legal|D\.?\s*L\.?)[:\s]*[A-Z]{1,2}[-\s]?\d{1,4}[-\s]?\d{2,4}',
        re.IGNORECASE
    )

    # Números de página aislados
    PAGE_NUMBER_PATTERN = re.compile(
        r'^[\s]*(?:Página|Pág\.?|Page|P\.?)?\s*\d{1,4}\s*(?:de\s*\d{1,4})?\s*$',
        re.IGNORECASE | re.MULTILINE
    )

    # Headers/Footers típicos
    HEADER_FOOTER_PATTERNS = [
        re.compile(r'^[\s]*(?:©|Copyright)\s*\d{4}.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*Todos\s*los\s*derechos\s*reservados.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*All\s*rights\s*reserved.*$', re.IGNORECASE | re.MULTILINE),
    ]

    # Branding/editorial específico (headers recurrentes).
    # Instrucción: generaliza eliminación de cabeceras/pies y texto promocional/guía
    # de academias (títulos de tema, marcas, emails, instrucciones de estudio, etc.).
    BRANDING_PATTERNS = [
        re.compile(r'^[\s]*Aspirantes\.es\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*Ingreso\s+al\s+Cuerpo\s+de\s+la\s+Guardia\s+Civil\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(
            r'^[\s]*Temario\s+de\s+Ingreso\s+al\s+Cuerpo\s+de\s+la\s+Guardia\s+Civil\.?\s*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*Temario\s+de\s+Ingreso\s+al\s+Cuerpo\s+de\s+la\s*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(r'^[\s]*Guardia\s+Civil\.?\s*$', re.IGNORECASE | re.MULTILINE),
        re.compile(
            r'^[\s]*TEMA\s*\d+\s*[\.-]?\s*GUARDIA\s*CIVIL\s*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*Tema\s*\d+\s*[\.-]?\s*Ingreso\s+al\s+Cuerpo\s+de\s+la\s+Guardia\s+Civil(?:\s*Aspirantes\.es)?\s*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*Tema\s*\d+.*Guardia\s+Civil.*Aspirantes\.es\s*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*.*Anabel,\s*Jorge,\s*Cristina,\s*Beli,\s*Carmen,.*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*.*Juan,\s*Pablo\s+y\s*Miguel.*$',
            re.IGNORECASE | re.MULTILINE
        ),
        re.compile(
            r'^[\s]*Por\s+el\s+tiempo\s+que\s+os\s+he\s+robado.*$',
            re.IGNORECASE | re.MULTILINE
        ),
    ]

    # Instrucciones de estudio / guías no sustantivas
    INSTRUCTIONAL_PATTERNS = [
        re.compile(r'^[\s]*Horas\s+de\s+Estudio.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*Test\s+de\s+Verificaci[oó]n\s+de\s+Nivel.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*Al\s+final\s+del\s+Tema.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*Controle\s+el\s+tiempo\s+real\s+de\s+estudio.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*En\s+el\s+cuadro\s+superior\s+de\s+observaciones.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*dudosas\s+que\s+deber[áa].*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*1ª\s+vuelta.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*2ª\s+vuelta.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*3ª\s+vuelta.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*hoja\s+de\s+respuestas.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*GU[IÍ]A\s+DEL\s+CIUDADANO.*$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*CCN-CERT.*$', re.IGNORECASE | re.MULTILINE),
    ]

    # Patrones de índice/TOC
    TOC_PATTERNS = [
        re.compile(r'^[\s]*(?:ÍNDICE|INDICE|CONTENIDO|TABLA\s*DE\s*CONTENIDO|INDEX|TABLE\s*OF\s*CONTENTS)[\s]*$',
                   re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*\d+\.[\d\.]*\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ\s]+\.{2,}\s*\d+\s*$', re.MULTILINE),  # Formato típico TOC
        re.compile(r'^[\s]*(?:Capítulo|Chapter|Sección|Section)\s+\d+.*\.{2,}\s*\d+\s*$', re.IGNORECASE | re.MULTILINE),
    ]

    # Información editorial
    EDITORIAL_PATTERNS = [
        re.compile(r'(?:Editorial|Publisher|Edita)[:\s]+[A-ZÁÉÍÓÚÑ][^\n]+', re.IGNORECASE),
        re.compile(r'(?:Impreso\s*en|Printed\s*in)[:\s]*[A-ZÁÉÍÓÚÑ][^\n]+', re.IGNORECASE),
        re.compile(r'(?:Primera|Segunda|Tercera|\d+[ªº]?)\s*(?:edición|edition)', re.IGNORECASE),
    ]

    # Información de autor en formato bibliográfico
    AUTHOR_PATTERNS = [
        re.compile(r'^[\s]*(?:Autor(?:es)?|Author(?:s)?)[:\s]+[^\n]+$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*(?:Coordinador(?:es)?|Editor(?:es)?)[:\s]+[^\n]+$', re.IGNORECASE | re.MULTILINE),
        re.compile(r'^[\s]*(?:Dirección|Director)[:\s]+[^\n]+$', re.IGNORECASE | re.MULTILINE),
    ]

    def __init__(self, min_substantive_ratio: float = 0.3):
        """Inicializa el filtro.

        Args:
            min_substantive_ratio: Ratio mínimo de contenido sustantivo (0-1).
                                   Chunks con ratio inferior se marcan como no sustantivos.
        """
        self.min_substantive_ratio = min_substantive_ratio

    def filter_chunk(self, content: str) -> MetadataFilterResult:
        """Filtra metadatos de un chunk y evalúa si es sustantivo.

        Args:
            content: Contenido del chunk a filtrar.

        Returns:
            MetadataFilterResult con el contenido limpio y métricas.
        """
        if not content or not content.strip():
            return MetadataFilterResult(
                original_content=content,
                clean_content="",
                is_substantive=False,
                substantive_ratio=0.0,
                filter_reasons=["Contenido vacío"]
            )

        detected_metadata = []
        clean_content = content

        # 1. Detectar y eliminar emails
        emails = self.EMAIL_PATTERN.findall(clean_content)
        if emails:
            detected_metadata.append({"type": "email", "values": emails})
            clean_content = self.EMAIL_PATTERN.sub('', clean_content)

        # 2. Detectar y eliminar URLs
        urls = self.URL_PATTERN.findall(clean_content)
        if urls:
            detected_metadata.append({"type": "url", "values": urls})
            clean_content = self.URL_PATTERN.sub('', clean_content)

        # 3. Detectar y eliminar teléfonos
        phones = self.PHONE_PATTERN.findall(clean_content)
        if phones:
            detected_metadata.append({"type": "phone", "values": phones})
            clean_content = self.PHONE_PATTERN.sub('', clean_content)

        # 4. Detectar y eliminar ISBN
        isbns = self.ISBN_PATTERN.findall(clean_content)
        if isbns:
            detected_metadata.append({"type": "isbn", "values": isbns})
            clean_content = self.ISBN_PATTERN.sub('', clean_content)

        # 5. Detectar y eliminar Depósito Legal
        depositos = self.DEPOSITO_LEGAL_PATTERN.findall(clean_content)
        if depositos:
            detected_metadata.append({"type": "deposito_legal", "values": depositos})
            clean_content = self.DEPOSITO_LEGAL_PATTERN.sub('', clean_content)

        # 6. Detectar y eliminar números de página aislados
        clean_content = self.PAGE_NUMBER_PATTERN.sub('', clean_content)

        # 7. Detectar y eliminar headers/footers
        for pattern in self.HEADER_FOOTER_PATTERNS:
            matches = pattern.findall(clean_content)
            if matches:
                detected_metadata.append({"type": "header_footer", "values": matches})
                clean_content = pattern.sub('', clean_content)

        # 7b. Detectar y eliminar branding/editorial específico
        for pattern in self.BRANDING_PATTERNS:
            matches = pattern.findall(clean_content)
            if matches:
                detected_metadata.append({"type": "branding", "values": matches})
                clean_content = pattern.sub('', clean_content)

        # 7c. Detectar y eliminar instrucciones/guías no sustantivas
        for pattern in self.INSTRUCTIONAL_PATTERNS:
            matches = pattern.findall(clean_content)
            if matches:
                detected_metadata.append({"type": "instructional", "values": matches})
                clean_content = pattern.sub('', clean_content)

        # 8. Detectar y eliminar información editorial
        for pattern in self.EDITORIAL_PATTERNS:
            matches = pattern.findall(clean_content)
            if matches:
                detected_metadata.append({"type": "editorial_info", "values": matches})
                clean_content = pattern.sub('', clean_content)

        # 9. Detectar y eliminar información de autor
        for pattern in self.AUTHOR_PATTERNS:
            matches = pattern.findall(clean_content)
            if matches:
                detected_metadata.append({"type": "author_info", "values": matches})
                clean_content = pattern.sub('', clean_content)

        # 10. Evaluar si es TOC/índice
        is_toc = self._is_table_of_contents(content)

        # Limpiar espacios múltiples y líneas vacías
        clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)
        clean_content = re.sub(r' {2,}', ' ', clean_content)
        clean_content = clean_content.strip()

        # Calcular ratio sustantivo
        original_len = len(content.strip())
        clean_len = len(clean_content)

        if original_len > 0:
            substantive_ratio = clean_len / original_len
        else:
            substantive_ratio = 0.0

        # Determinar razones de filtrado
        filter_reasons = []

        if is_toc:
            filter_reasons.append("Detectado como índice/TOC")
            substantive_ratio = 0.0  # TOC siempre es no sustantivo

        if detected_metadata:
            types = [m["type"] for m in detected_metadata]
            filter_reasons.append(f"Metadatos detectados: {', '.join(set(types))}")

        if substantive_ratio < self.min_substantive_ratio:
            filter_reasons.append(f"Ratio sustantivo bajo: {substantive_ratio:.2%}")

        # Determinar si es sustantivo
        is_substantive = (
            substantive_ratio >= self.min_substantive_ratio
            and not is_toc
            and len(clean_content) >= 50  # Mínimo 50 caracteres útiles
        )

        return MetadataFilterResult(
            original_content=content,
            clean_content=clean_content,
            is_substantive=is_substantive,
            substantive_ratio=substantive_ratio,
            detected_metadata=detected_metadata,
            filter_reasons=filter_reasons
        )

    def _is_table_of_contents(self, content: str) -> bool:
        """Detecta si el contenido es un índice/tabla de contenidos.

        Args:
            content: Texto a analizar.

        Returns:
            True si parece ser un TOC.
        """
        # Verificar patrones de TOC
        for pattern in self.TOC_PATTERNS:
            if pattern.search(content):
                return True

        # Heurística: muchas líneas con formato "texto ... número"
        lines = content.split('\n')
        toc_like_lines = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Patrón: texto seguido de puntos y número de página
            if re.search(r'\.{3,}\s*\d+\s*$', line):
                toc_like_lines += 1

            # Patrón: número de sección seguido de texto
            if re.match(r'^\d+\.[\d\.]*\s+[A-ZÁÉÍÓÚÑ]', line):
                toc_like_lines += 1

        # Si más del 50% de las líneas no vacías parecen TOC
        non_empty_lines = len([l for l in lines if l.strip()])
        if non_empty_lines > 3 and toc_like_lines / non_empty_lines > 0.5:
            return True

        return False

    def batch_filter(self, chunks: List[str]) -> List[MetadataFilterResult]:
        """Filtra múltiples chunks.

        Args:
            chunks: Lista de contenidos de chunks.

        Returns:
            Lista de MetadataFilterResult para cada chunk.
        """
        return [self.filter_chunk(chunk) for chunk in chunks]

    def get_substantive_chunks(
        self,
        chunks: List[str]
    ) -> Tuple[List[str], List[int], List[int]]:
        """Filtra chunks y devuelve solo los sustantivos.

        Args:
            chunks: Lista de contenidos de chunks.

        Returns:
            Tuple de:
            - Lista de contenidos limpios sustantivos
            - Lista de índices de chunks sustantivos
            - Lista de índices de chunks no sustantivos
        """
        results = self.batch_filter(chunks)

        substantive_contents = []
        substantive_indices = []
        non_substantive_indices = []

        for i, result in enumerate(results):
            if result.is_substantive:
                substantive_contents.append(result.clean_content)
                substantive_indices.append(i)
            else:
                non_substantive_indices.append(i)

        return substantive_contents, substantive_indices, non_substantive_indices


def filter_metadata_from_chunk(content: str, min_ratio: float = 0.3) -> MetadataFilterResult:
    """Función helper para filtrar metadatos de un chunk.

    Args:
        content: Contenido del chunk.
        min_ratio: Ratio mínimo de contenido sustantivo.

    Returns:
        MetadataFilterResult con el análisis.
    """
    filter_instance = PDFMetadataFilter(min_substantive_ratio=min_ratio)
    return filter_instance.filter_chunk(content)


# ==========================================
# EJEMPLO DE USO
# ==========================================

if __name__ == "__main__":
    # Ejemplo de uso
    test_chunks = [
        # Chunk con metadatos
        """
        Manual de Derecho Penal
        Autor: Dr. Juan García López
        Email: jgarcia@universidad.es
        Editorial Jurídica, 2024
        ISBN: 978-84-1234-567-8

        Artículo 138. El que matare a otro será castigado,
        como reo de homicidio, con la pena de prisión de
        diez a quince años.
        """,

        # Chunk que es índice
        """
        ÍNDICE

        Capítulo 1. Introducción .................. 5
        Capítulo 2. Delitos contra la vida ....... 15
        2.1. Homicidio ........................... 17
        2.2. Asesinato ........................... 23
        Capítulo 3. Delitos patrimoniales ....... 45
        """,

        # Chunk sustantivo
        """
        Artículo 139 del Código Penal:

        Será castigado con la pena de prisión de quince a veinticinco años,
        como reo de asesinato, el que matare a otro concurriendo alguna
        de las circunstancias siguientes:

        1.ª Con alevosía.
        2.ª Por precio, recompensa o promesa.
        3.ª Con ensañamiento, aumentando deliberada e inhumanamente
            el dolor del ofendido.
        """
    ]

    filter_instance = PDFMetadataFilter()

    for i, chunk in enumerate(test_chunks):
        print(f"\n{'='*60}")
        print(f"CHUNK {i + 1}")
        print('='*60)

        result = filter_instance.filter_chunk(chunk)

        print(f"Es sustantivo: {result.is_substantive}")
        print(f"Ratio sustantivo: {result.substantive_ratio:.2%}")
        print(f"Razones de filtrado: {result.filter_reasons}")

        if result.detected_metadata:
            print(f"Metadatos detectados:")
            for meta in result.detected_metadata:
                print(f"  - {meta['type']}: {meta['values']}")

        print(f"\nContenido limpio ({len(result.clean_content)} chars):")
        print(result.clean_content[:200] + "..." if len(result.clean_content) > 200 else result.clean_content)
