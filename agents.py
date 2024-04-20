from crewai import Agent
from tools import MemoryTools

class LTMAgents():
    def context_agent(self):
        return Agent(
            role="Context Specialist",
            goal='Decides whether to store something from one user comment in the database.',
            tools=[MemoryTools.insert_memory],
            backstory="""
            You specialize in understanding the context of a conversation between a human and an AI.
            You are part of a team building a knowledge base to personalize communication with a human user.
            Your role is to decide what to extract from a user message and insert it into the database accordingly by:
            - Analyzing the given MESSAGE and deciding whether it contains advice and an associated task.
            - Analyzing the given MESSAGE and deciding whether it contains meaningful information.
            """,
            verbose=True,
            max_iter=3,
            allow_delegation=False
        )
    
    def retrieval_agent(self):
        return Agent(
            role="Retrieval Specialist",
            goal='Determines relevant memories to retrieve based on the message context.',
            tools=[MemoryTools.retrieve_memories, MemoryTools.concatenate_memo_texts],
            backstory="""
            You specialize in retrieving relevant information from memory based on the most recent user message.
            You are part of a team building a knowledge base to personalize communication with a human user.
            Your role is to analyze the given MESSAGE and decide whether to retrieve relevant memos from the database.
            This will be used to inform the AI on applicable properties, preferences and restrictions the human may have.
            """,
            verbose=True,
            max_iter=3,
            allow_delegation=False
        )