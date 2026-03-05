"""QuestionOption model - Opciones de respuesta para preguntas."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class QuestionOption(BaseModel):
    """Modelo de opción de respuesta para una pregunta.

    Compatible con la estructura de Supabase:
    - Cada opción es una fila independiente en la tabla question_options
    - Se relaciona con questions mediante question_id
    - Permite flexibilidad en el número de opciones (3, 4, 5, etc.)
    """
    id: Optional[int] = None
    question_id: int
    answer: str
    is_correct: bool = False
    option_order: int  # 1, 2, 3, 4, etc.
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)

    def to_dict(self, exclude_none: bool = False):
        """Convierte a dict para SQLite."""
        data = self.model_dump(exclude_none=exclude_none)
        # Convertir datetime a string para SQLite
        if data.get('created_at'):
            data['created_at'] = data['created_at'].isoformat()
        if data.get('updated_at'):
            data['updated_at'] = data['updated_at'].isoformat()
        return data

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }