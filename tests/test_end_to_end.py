"""
Test de punta a punta del orquestador REAL (grafo, Supervisor, agentes
ReAct creados con create_react_agent, y herramientas reales) usando un
chat model FALSO y guionado, sin necesitar ningún LLM ni API key.

El modelo falso, además, hace que el Supervisor falle igual que
llama3.2 en la práctica (with_structured_output devuelve None), para
verificar que el flujo completo se sostiene gracias al plan B por reglas.
"""

from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

import graph
import supervisor

COMENTARIOS = [
    "El nuevo checkout es mucho más rápido, me encantó la experiencia.",
    "Tarda demasiado en cargar el paso de pago, es frustrante.",
    "Tuve un error al confirmar la compra, muy mala experiencia.",
]
RATINGS = [5, 2, 1]


class _ChatGuionado(BaseChatModel):
    """Chat model falso que responde según el rol indicado en el system prompt."""

    @property
    def _llm_type(self) -> str:
        return "chat-guionado"

    def bind_tools(self, tools, **kwargs):  # create_react_agent lo necesita
        return self

    def with_structured_output(self, _esquema, **kwargs):
        # Simula el fallo real de llama3.2: no devuelve una decisión.
        return RunnableLambda(lambda _mensajes: None)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system = str(messages[0].content) if messages else ""
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

        if "Agente de Investigación" in system:
            if not tool_messages:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[{"name": "buscar_comentarios", "args": {"tema": "checkout"}, "id": "c1"}],
                )
            else:
                respuesta = AIMessage(
                    content=f"Encontré 3 comentarios. Textos: {COMENTARIOS}. Ratings: {RATINGS}."
                )

        elif "Agente de Análisis" in system:
            if len(tool_messages) == 0:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "analizar_sentimiento", "args": {"comentarios": COMENTARIOS}, "id": "a1"}
                    ],
                )
            elif len(tool_messages) == 1:
                respuesta = AIMessage(
                    content="",
                    tool_calls=[{"name": "calcular_estadisticas", "args": {"numeros": RATINGS}, "id": "a2"}],
                )
            else:
                respuesta = AIMessage(
                    content="Sentimiento mixto (1 positivo, 2 negativos). Rating promedio: 2.67."
                )

        else:  # síntesis final del Supervisor
            respuesta = AIMessage(content="Síntesis: el checkout tiene sentimiento mixto y rating 2.67.")

        return ChatResult(generations=[ChatGeneration(message=respuesta)])


def test_flujo_real_completo_con_supervisor_que_falla(monkeypatch):
    modelo = _ChatGuionado()
    monkeypatch.setattr(supervisor, "get_llm", lambda: modelo)
    monkeypatch.setattr(graph, "get_llm", lambda: modelo)

    grafo = graph.build_graph()
    resultado = grafo.invoke(
        {
            "messages": [HumanMessage(content="Analizá qué opinan los usuarios del checkout.")],
            "contribuciones": {},
            "pasos": 0,
        },
        config={"recursion_limit": 25},
    )

    # Orden exacto de la delegación en la traza
    agentes = [getattr(m, "name", None) or "usuario" for m in resultado["messages"]]
    assert agentes == [
        "usuario",
        "supervisor",     # delega en el investigador
        "investigador",
        "supervisor",     # delega en el analista
        "analista",
        "supervisor",     # síntesis final
    ]

    # Ambos especialistas aportaron, y con datos reales de sus herramientas
    assert set(resultado["contribuciones"]) == {"investigador", "analista"}
    assert "Ratings" in resultado["contribuciones"]["investigador"]
    assert "2.67" in resultado["contribuciones"]["analista"]

    # 3 vueltas del Supervisor: investigador, analista, cierre
    assert resultado["pasos"] == 3
    assert "Síntesis" in resultado["messages"][-1].content
