from crewai import Agent
from tools import MemoryTools

class LTMAgents():
    def context_agent(self):
        return Agent(
            role="Context Specialist",
            goal='Decides whether to store something from one user comment in the DB.',
            tools=[],
            backstory="""
            Your job is to assess a most recent message provided in a chat between AI and a user in order to determine if the conversation contains any details worth remembering for later. 
            You are part of a team building a knowledge base to personalize communication with the user.
            You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.
            Treat the user as a programatic object and take note of any properties (ex. Preferences, Fears, Likes, Dislikes, Problems etc.)
            When you receive a message, you perform a sequence of steps consisting of:

            1. Analyze the message for information.
            2. If it has any information worth recording.

            You should only respond with the type of property and the value (ex. Dislike: Bananas). Absolutely no other information should be provided.
            Take a deep breath, think step by step, and then analyze the message.
            """,
            verbose=True,
            max_iter=3,
            allow_delegation=False
        )

    def memory_agent(self):
        return Agent(
            role="Memory Specialist",
            goal='Decides whether to store something from one user comment in the DB.',
            tools=[MemoryTools.insert_memory],
            backstory="""
            You will receive a memory from the context_specialist agent that they considered should be stored in the DB.
            Your role is to commit the memory to the database utilizing the insert_memory tool in MemoryTools.

            Take the property type and the property value provided by the context_specialist and store it in the memory store utilizing the MemoryTools provided.
            Pass in the memory store object into the insert_memory tool and insert the memory into the database.
            Take a deep breath, think step by step and store the memories in the database.
            """,
            verbose=True,
            max_iter=3,
            allow_delegation=False
        )