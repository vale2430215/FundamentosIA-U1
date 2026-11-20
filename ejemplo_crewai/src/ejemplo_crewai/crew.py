from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from crewai.llm import LLM
from crewai_tools import SerperDevTool

search_tool = SerperDevTool(search_type="news", n_results=10)

llm_llama3 = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)
llm_phi4 = LLM(
    model="ollama/phi4-mini",
    base_url="http://localhost:11434"
)

@CrewBase
class EjemploCrewai():
    """EjemploCrewai crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def investigador_noticias(self) -> Agent:
        return Agent(
            config=self.agents_config['investigador_noticias'],
            verbose=True,
            llm=llm_llama3,
            tools=[search_tool]
        )

    @agent
    def redactor_noticias(self) -> Agent:
        return Agent(
            config=self.agents_config['redactor_noticias'],
            verbose=True,
            llm=llm_llama3  
        )

    @task
    def tarea_investigacion(self) -> Task:
        return Task(config=self.tasks_config['tarea_investigacion'])

    @task
    def tarea_redaccion(self) -> Task:
        return Task(config=self.tasks_config['tarea_redaccion'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            tracing=False,
        )
