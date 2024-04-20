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
            max_iter=2,
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
            max_iter=5,
            allow_delegation=False
        )
    
    def user_proxy_agent(self):
        return Agent(
            role="Nutrition Specialist",
            goal='Provide the best meal plan based on the preferences and restrictions of the user.',
            tools=[],
            backstory="""
            You are a helpful assistant. You specialize in providing nutrition advice and meal plans. You are very cognizant of the user's
            preferences and restrictions. If context and personalized information is provided to you by the Retrieval Specialist, use it
            to respond to the best of your ability to the user.
            """,
            verbose=True,
            max_iter=5,
            allow_delegation=False
        )