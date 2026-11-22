from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.llm import LLM
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()
llm_llama3 = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)
llm_phi4 = LLM(
    model="ollama/phi4-mini",
    base_url="http://localhost:11434"
)

@CrewBase
class TallerCrewai():
    """TallerCrewai crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    @agent
    def investigador_tecnologias(self) -> Agent:
        return Agent(
            config=self.agents_config['investigador_tecnologias'],
            verbose=True,
            tools=[search_tool]
        )

    @agent
    def redactor_informe(self) -> Agent:
        return Agent(
            config=self.agents_config['redactor_informe'],
            verbose=True
        )

    @task
    def tarea_investigacion(self) -> Task:
        return Task(
            config=self.tasks_config['tarea_investigacion'],
        )

    @task
    def tarea_redaccion(self) -> Task:
        return Task(
            config=self.tasks_config['tarea_redaccion'],
            output_file='informe.md'
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )