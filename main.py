"""
Script de demostración del orquestador multi-agente.

Lanza una consulta que obliga al Supervisor a delegar primero al
Agente de Investigación y después al Agente de Análisis antes de poder
cerrar la tarea, y guarda la traza completa de la delegación en
/traces (mensaje por mensaje, con el agente responsable de cada uno).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from graph import build_graph

load_dotenv()

TRACES_DIR = Path(__file__).parent / "traces"
TRACES_DIR.mkdir(exist_ok=True)

CONSULTA_DEMO = (
    "Necesito saber qué opinan los usuarios sobre el checkout: "
    "buscá sus comentarios, analizá el sentimiento general y calculá "
    "el rating promedio."
)


def _mensaje_a_dict(mensaje) -> dict:
    """Convierte un mensaje de LangChain a un dict serializable para la traza."""
    return {
        "agente": getattr(mensaje, "name", None) or "usuario",
        "tipo": mensaje.__class__.__name__,
        "contenido": mensaje.content,
    }


def _guardar_traza(nombre_archivo: str, mensajes: list, contribuciones: dict, pasos: int) -> None:
    traza = {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pasos_del_supervisor": pasos,
        "contribuciones": contribuciones,
        "mensajes": [_mensaje_a_dict(m) for m in mensajes],
    }
    path = TRACES_DIR / f"{nombre_archivo}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(traza, f, indent=2, ensure_ascii=False)
    print(f"✓ Traza guardada en {path.relative_to(Path(__file__).parent)}")


def main() -> None:
    provider = os.getenv("GENERATION_PROVIDER", "ollama")

    print(f"\n{'='*70}")
    print("Orquestador Multi-Agente — Demo de Delegación")
    print(f"{'='*70}")
    print(f"Proveedor: {provider}")

    if provider == "ollama":
        print(
            "\n⚠ Asegurate de tener Ollama corriendo localmente y el modelo "
            "descargado (ver README.md)."
        )

    print(f"\nConsulta: {CONSULTA_DEMO}\n")

    grafo = build_graph()

    resultado = grafo.invoke(
        {
            "messages": [HumanMessage(content=CONSULTA_DEMO)],
            "contribuciones": {},
            "pasos": 0,
        },
        config={"recursion_limit": 25},
    )

    print(f"{'='*70}")
    print("Traza de la delegación")
    print(f"{'='*70}\n")

    for m in resultado["messages"]:
        agente = getattr(m, "name", None) or "usuario"
        print(f"[{agente}] {m.content}\n")

    print(f"{'='*70}")
    print(f"Pasos totales del Supervisor: {resultado['pasos']}")
    print(f"Contribuciones registradas: {list(resultado['contribuciones'].keys())}")
    print(f"{'='*70}\n")

    _guardar_traza(
        "trace_demo_delegacion",
        resultado["messages"],
        resultado["contribuciones"],
        resultado["pasos"],
    )


if __name__ == "__main__":
    main()
