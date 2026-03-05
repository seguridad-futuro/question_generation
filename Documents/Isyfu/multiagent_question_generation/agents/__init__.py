"""Agents module - Multi-agent question generation system.

Este módulo contiene agentes autónomos con arquitectura StateGraph moderna:

- Agente Z (agent_z_rewriter): Genera chunks coherentes con análisis de estructura
- Agente B (agent_b_generator): Genera preguntas usando herramientas y razonamiento
- Agente C (agent_c_evaluator): Evalúa calidad con Ragas y decide aprobar/rechazar/retry
- Agente E (agent_e_pdf_generator): Genera PDFs profesionales agrupados por tema
"""

from agents.agent_z_rewriter import RewriterAgent

from agents.agent_b_generator import (
    create_agent_b_generator,
    generate_questions_for_chunk,
    generate_questions_for_chunks
)

from agents.agent_c_evaluator import (
    create_agent_c_evaluator,
    evaluate_question
)

from agents.agent_e_pdf_generator import (
    PDFGeneratorAgent,
    generate_pdfs_from_questions
)

__all__ = [
    # Agente Z
    "RewriterAgent",
    # Agente B
    "create_agent_b_generator",
    "generate_questions_for_chunk",
    "generate_questions_for_chunks",
    # Agente C
    "create_agent_c_evaluator",
    "evaluate_question",
    # Agente E
    "PDFGeneratorAgent",
    "generate_pdfs_from_questions",
]
