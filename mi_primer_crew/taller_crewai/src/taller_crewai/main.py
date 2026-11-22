def run():
    inputs = {
        'topic': 'Inteligencia Artificial en la educación',
        'current_year': str(datetime.now().year),
    }
    try:
        TallerCrewai().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"Error al ejecutar: {e}")