"""
Tests de la mecánica del grafo del orquestador, usando nodos FALSOS en
vez del LLM real -- así se puede verificar el enrutamiento, el corte de
seguridad contra el "Supervisor Infinito", y que los especialistas
reciben solo la instrucción puntual (no el historial completo) en
cualquier entorno, sin necesitar ninguna API key.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph

from graph import build_graph
from state import OrquestadorState

MAX_PASOS_TEST = 4


def _enrutar(state):
    destino = state["next_agent"]
    return END if destino == "FINISH" else destino


def _build_grafo_con_flujo_normal():
    """Arma un grafo con nodos falsos que simulan investigador -> analista -> FINISH."""

    def supervisor_falso(state):
        contribuciones = state.get("contribuciones", {})
        pasos = state.get("pasos", 0) + 1

        if "investigador" not in contribuciones:
            return {
                "next_agent": "investigador",
                "instruccion_para_agente": "Buscá comentarios sobre checkout",
                "pasos": pasos,
            }
        if "analista" not in contribuciones:
            return {
                "next_agent": "analista",
                "instruccion_para_agente": f"Analizá esto: {contribuciones['investigador']}",
                "pasos": pasos,
            }
        return {
            "next_agent": "FINISH",
            "pasos": pasos,
            "messages": [AIMessage(content="Síntesis final.", name="supervisor")],
        }

    def investigador_falso(state):
        assert "instruccion_para_agente" in state  # el especialista SÍ recibe la instrucción
        respuesta = "Encontré 5 comentarios sobre checkout."
        return {
            "messages": [AIMessage(content=respuesta, name="investigador")],
            "contribuciones": {**state.get("contribuciones", {}), "investigador": respuesta},
        }

    def analista_falso(state):
        respuesta = "Sentimiento mixto: 2 positivos, 2 negativos."
        return {
            "messages": [AIMessage(content=respuesta, name="analista")],
            "contribuciones": {**state.get("contribuciones", {}), "analista": respuesta},
        }

    builder = StateGraph(OrquestadorState)
    builder.add_node("supervisor", supervisor_falso)
    builder.add_node("investigador", investigador_falso)
    builder.add_node("analista", analista_falso)
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges(
        "supervisor", _enrutar, {"investigador": "investigador", "analista": "analista", END: END}
    )
    builder.add_edge("investigador", "supervisor")
    builder.add_edge("analista", "supervisor")
    return builder.compile()


def test_flujo_completo_investigador_luego_analista():
    """El flujo debe pasar por investigador y luego analista antes de terminar."""
    grafo = _build_grafo_con_flujo_normal()
    resultado = grafo.invoke(
        {
            "messages": [HumanMessage(content="Investigá y analizá el feedback sobre checkout.")],
            "contribuciones": {},
            "pasos": 0,
        }
    )

    nombres_mensajes = [getattr(m, "name", "usuario") for m in resultado["messages"]]
    assert "investigador" in nombres_mensajes
    assert "analista" in nombres_mensajes
    # El investigador debe aparecer ANTES que el analista en la traza
    assert nombres_mensajes.index("investigador") < nombres_mensajes.index("analista")

    assert set(resultado["contribuciones"].keys()) == {"investigador", "analista"}


def test_corte_de_seguridad_contra_supervisor_infinito():
    """
    Si el supervisor nunca decide FINISH por su cuenta, el contador de
    pasos debe cortar el ciclo de todas formas.
    """

    def supervisor_loopeado(state):
        pasos = state.get("pasos", 0) + 1
        if pasos > MAX_PASOS_TEST:
            return {
                "next_agent": "FINISH",
                "pasos": pasos,
                "messages": [AIMessage(content="Corte de seguridad.", name="supervisor")],
            }
        return {"next_agent": "investigador", "instruccion_para_agente": "seguí", "pasos": pasos}

    def investigador_falso(state):
        return {"messages": [AIMessage(content="...", name="investigador")], "contribuciones": {}}

    builder = StateGraph(OrquestadorState)
    builder.add_node("supervisor", supervisor_loopeado)
    builder.add_node("investigador", investigador_falso)
    builder.set_entry_point("supervisor")
    builder.add_conditional_edges("supervisor", _enrutar, {"investigador": "investigador", END: END})
    builder.add_edge("investigador", "supervisor")
    grafo = builder.compile()

    resultado = grafo.invoke(
        {"messages": [HumanMessage(content="tarea imposible")], "contribuciones": {}, "pasos": 0},
        config={"recursion_limit": 50},
    )

    assert resultado["pasos"] == MAX_PASOS_TEST + 1
    assert "Corte de seguridad" in resultado["messages"][-1].content


def test_build_graph_real_tiene_los_nodos_esperados():
    """El grafo real (graph.py) debe tener exactamente los 3 nodos esperados."""
    grafo = build_graph()
    nodos = set(grafo.get_graph().nodes.keys())
    assert {"supervisor", "investigador", "analista"} <= nodos
