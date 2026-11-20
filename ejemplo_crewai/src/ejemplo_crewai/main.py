#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from ejemplo_crewai.crew import EjemploCrewai
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    inputs = {
        'topic': 'Taylor Swift',
        'current_year': str(datetime.now().year),
    }
    try:
        resultado = EjemploCrewai().crew().kickoff(inputs=inputs)
        texto = resultado.raw

        carpeta = Path(__file__).resolve().parents[2] / "noticias"
        carpeta.mkdir(exist_ok=True)

        nombre = f"noticia_{datetime.now():%Y-%m-%d_%H-%M}.md"
        archivo = carpeta / nombre
        archivo.write_text(texto, encoding="utf-8")

        print("\n" + "=" * 60)
        print(f"NOTICIA GENERADA ({archivo})")
        print("=" * 60 + "\n")
        print(texto)
    except Exception as e:
        raise Exception(f"Error al ejecutar: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs",
        'current_year': str(datetime.now().year)
    }
    try:
        EjemploCrewai().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        EjemploCrewai().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        EjemploCrewai().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

def run_with_trigger():
    """
    Run the crew with trigger payload.
    """
    import json

    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    inputs = {
        "crewai_trigger_payload": trigger_payload,
        "topic": "",
        "current_year": ""
    }

    try:
        result = EjemploCrewai().crew().kickoff(inputs=inputs)
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the crew with trigger: {e}")
