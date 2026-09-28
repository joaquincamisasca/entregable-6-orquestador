"""
Esquema de Estado compartido del orquestador.

Hereda de MessagesState (la misma base que usamos en el Entregable 5)
y le agrega los campos extra que necesita la topología jerárquica:

- next_agent: a quién le toca actuar ahora, según decide el Supervisor.
- instruccion_para_agente: la instrucción puntual y acotada que el
  Supervisor le da al especialista elegido. Existe para evitar la
  "Contaminación de Contexto" que advierte la consigna: en vez de que
  cada especialista reciba TODO el historial de mensajes del sistema,
  recibe solo esta instrucción específica.
- contribuciones: rastrea qué aportó cada agente, para que el Supervisor
  (y cualquiera que audite la traza) pueda ver de un vistazo qué
  información ya se recolectó y qué falta.
- pasos: contador de vueltas del Supervisor. Sirve como salvavidas
  contra el "Supervisor Infinito" (un bucle de correcciones eternas
  entre supervisor y especialistas sin converger nunca).
"""

from typing import Dict, Literal

from langgraph.graph import MessagesState

NombreAgente = Literal["investigador", "analista", "FINISH"]


class OrquestadorState(MessagesState):
    """Estado compartido entre el Supervisor y los agentes especialistas."""

    next_agent: NombreAgente
    instruccion_para_agente: str
    contribuciones: Dict[str, str]
    pasos: int
