from crewai import Task

class MemoryTasks():
    def determine_memory_task(self, agent, user_comment):
        return Task(
            description=f"""
            Determine if the user_comment contains a memory that the user might forgot. 
            Consider how would they ask you for this information. This will be the question
            Copy the information from the TEXT that should be committed to memory. This will be the answer

            User Comment: {user_comment}
            """,
            expected_output=f"""The question and answer combination""",
            agent=agent,
            async_execution=True
    )

    def store_memory(self, agent, database):
        return Task(
            description=f"""
            Retrieve the question and answer from the context_agent.
            Store the question and answer in the {database} utilizing the insert_memory tool in MemoryTools
            """,
            expected_output=f"""Confirmation that the memory was stored""",
            agent=agent,
            async_execution=True
    )


  

