from crewai import Task

class MemoryTasks():
    def determine_memory_task(self, agent, user_comment):
        return Task(
            description=f"""
            Determine most recent message provided in a chat between AI and a user in order to determine if the conversation contains any details worth remembering for later. 
            Perform a sequence of steps consisting of:

                1. Analyze the message for information.
                2. If it has any information worth recording.

            You should only respond with the type of property and the value (ex. Dislike: Bananas). Absolutely no other information should be provided.

            Most Recent Message: {user_comment}
            """,
            expected_output=f"""The property and value combination""",
            agent=agent,
            async_execution=False
    )

    def store_memory(self, agent):
        return Task(
            description=f"""
            Commit the memory provided by the context_agent into the DB utilizing the insert_memory tool in MemoryTools.

            Take the property type and the property value provided by the context_specialist and store it in the memory store utilizing the MemoryTools provided.
            Take a deep breath, think step by step and store the memories in the database.
            """,
            expected_output=f"""The memory stored in the database.""",
            agent=agent,
            async_execution=False
    )


  

