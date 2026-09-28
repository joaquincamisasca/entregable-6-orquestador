"""
Grafo principal del orquestador: topología jerárquica (Supervisor +
Especialistas).

    START -> supervisor --(next_agent)--> investigador -> supervisor -> ...
                  |                            analista -> supervisor -> ...
                  +----------------------> FINISH -> END

El Supervisor decide, en cada vuelta, a qué especialista delegar (o si
ya se puede cerrar). Cada especialista es un sub-agente ReAct completo
(create_react_agent), invocado con SOLO la instrucción puntual que le
dio el Supervisor — nunca con el historial completo de mensajes del
sistema. Esto es la mitigación explícita de la "Contaminación de
Contexto" que advierte la consigna.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from agents.analyst_agent import crear_agente_analista
from agents.research_agent import crear_agente_investigador
from state import OrquestadorState
from supervisor import get_llm, nodo_supervisor


def _nodo_investigador(state: OrquestadorState) -> dict:
    """
    Ejecuta al Agente de Investigación con SOLO la instrucción puntual
    del Supervisor (no todo el historial), y registra su contribución.
    """
    llm = get_llm()
    agente = crear_agente_investigador(llm)

    instruccion = state["instruccion_para_agente"]
    resultado = agente.invoke({"messages": [HumanMessage(content=instruccion)]})
    respuesta = resultado["messages"][-1].content

    contribuciones = {**state.get("contribuciones", {}), "investigador": respuesta}

    return {
        "messages": [AIMessage(content=respuesta, name="investigador")],
        "contribuciones": contribuciones,
    }


def _nodo_analista(state: OrquestadorState) -> dict:
    """
    Ejecuta al Agente de Análisis con SOLO la instrucción puntual del
    Supervisor (que normalmente incluye los datos recolectados por el
    investigador, ya extraídos), y registra su contribución.
    """
    llm = get_llm()
    agente = crear_agente_analista(llm)

    instruccion = state["instruccion_para_agente"]
    resultado = agente.invoke({"messages": [HumanMessage(content=instruccion)]})
    respuesta = resultado["messages"][-1].content

    contribuciones = {**state.get("contribuciones", {}), "analista": respuesta}

    return {
        "messages": [AIMessage(content=respuesta, name="analista")],
        "contribuciones": contribuciones,
    }


def _enrutar(state: OrquestadorState) -> str:
    """
    Arista condicional desde el Supervisor: traduce next_agent (definido
    con Literal en el estado) al nombre del nodo destino, o a END.
    """
    destino = state["next_agent"]
    if destino == "FINISH":
        return END
    return destino


def build_graph():
    """
    Arma y compila el StateGraph completo del orquestador.

    Returns:
        Grafo compilado, listo para invocar con .invoke()
    """
    builder = StateGraph(OrquestadorState)

    builder.add_node("supervisor", nodo_supervisor)
    builder.add_node("investigador", _nodo_investigador)
    builder.add_node("analista", _nodo_analista)

    builder.set_entry_point("supervisor")

    builder.add_conditional_edges(
        "supervisor",
        _enrutar,
        {"investigador": "investigador", "analista": "analista", END: END},
    )

    # Después de cada especialista, siempre volvemos al Supervisor para
    # que decida el próximo paso (o cierre la tarea).
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")

    return builder.compile()
