from crewai import Agent
from tools import MemoryTools

class LTMAgents():
    def context_agent(self):
      return Agent(
        role="Context Specialist",
        goal='Decides whether to store something from one user comment in the DB.',
        tools=[],
        backstory="""
        You are a context specialist. 
        You will be provided a comment by the user. 
        Imagine that the user forgot this information in the TEXT. 
        Consider how would they ask you for this information.
        Copy the information from the TEXT that should be committed to memory. 
        Include no other text in your response.
        Add no explanation.
        """,
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )

    def memory_agent(self):
      return Agent(
        role="Memory Specialist",
        goal='Decides whether to store something from one user comment in the DB.',
        tools=MemoryTools.tools(),
        backstory="""
        You will receive a memory from the context_specialist agent that they considered should be stored in the DB.
        Your role is to commit the memory to the DB utilizing the insert_memory tool in MemoryTools.
        """,
        verbose=True,
        max_iter=3,
        allow_delegation=False
    )