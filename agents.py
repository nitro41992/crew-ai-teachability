from crewai import Agent
from tools import MemoryTools

class LTMAgents():
    def context_agent(self):
        return Agent(
            role="Context Specialist",
            goal='Decides whether to store something from one user comment in the database.',
            tools=[],
            backstory="""
            Your job is to assess the most recent message provided in a chat between AI and a user in order to determine if the conversation contains any details worth remembering for later. 
            You are part of a team building a knowledge base to personalize communication with the user.
            You play the critical role of assessing the message to determine if it contains any information worth recording in the knowledge base.
            Treat the user as a programatic object and take note of any properties (ex. Preferences, Fears, Likes, Dislikes, Problems etc.)
            When you receive a message, you perform a sequence of steps consisting of:

            1. Analyze the message for information.
            2. If it has any information worth recording.

            You should only respond with the type of property and the value (ex. Dislike: Bananas). 
            If there are multiple properties, make sure to provide absolutely all memories as a list.
            Take a deep breath, think step by step, and then analyze the message.
            """,
            verbose=True,
            max_iter=5,
            allow_delegation=False
        )

    def memory_agent(self):
        return Agent(
            role="Memory Specialist",
            goal='Decides whether to store something from one user comment in the database.',
            tools=[MemoryTools.insert_memory],
            backstory="""
            You will receive memories from the context_specialist agent that they considered should be stored in the database.
            Your role is to commit the memories to the database utilizing the insert_memory tool in MemoryTools.
            If muliple memories are provided, insert each into the database.

            Check with the retrieval_agent to make sure that the memory does not already exist. Do not insert it into the database if it already exists.

            Take the property type and the property value for each memory provided by the context_specialist and store it in the memory store utilizing the MemoryTools provided.
            Pass in the memory store object into the insert_memory tool and insert the memories into the database.
            """,
            verbose=True,
            max_iter=5,
            allow_delegation=True
        )
    
    def retrieval_agent(self):
        return Agent(
            role="Retrieval Specialist",
            goal='Determines relevant memories to retrieve from the database based on the message context.',
            tools=[MemoryTools.retrieve_memories, MemoryTools.concatenate_memo_texts],
            backstory="""
            Your job is to assess the most recent message provided in a chat between AI and a user in order to retrieve relevant memories to consider
            that are relevant to the context of the message.

            Utilize the context of the current user message to retrieve and combine relevant memories.
            Each memory will have a memory and memory type. Your job is to retrieve both values
            This will be used to inform the AI on applicable preferences and restrictions the user may have.

            Take the context and retrieve and combine the relevant memories utilizing the retrieve_memories and _concatenate_memo_texts MemoryTools provided.
            You should retrieve the memory and type of memory.
            If no relevant memories are retrieved, only respond with No Memories Found
            If memories are retrieved, provide the combined memories as a succinct bulleted list of of the user's properties, preferences and restrictions.
            """,
            verbose=True,
            max_iter=5,
            allow_delegation=True
        )