from dotenv import load_dotenv
from crewai import Crew
from tasks import MemoryTasks
from agents import LTMAgents


def main():
    load_dotenv()
    
    print("## Chat Test")
    print('-------------------------------')
    user_context = input("I want to create a meal plan for myself for lunch but Im allergic to fish?\n")

    tasks = MemoryTasks()
    agents = LTMAgents()
    
    # create agents
    context_agent = agents.context_agent()
    memory_agent = agents.memory_agent()
    
    # create tasks
    context_task = tasks.determine_memory_task(context_agent, user_context)
    memory_task = tasks.store_memory(memory_agent)
    
    memory_task.context = [context_task]
    
    crew = Crew(
      agents=[
          context_agent, 
          memory_agent
      ],
      tasks=[
        context_task,
        memory_task
      ]
    )
    
    result = crew.kickoff()
    
    print(result)
    
if __name__ == "__main__":
    main()