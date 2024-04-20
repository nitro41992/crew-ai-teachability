from crewai import Task

class MemoryTasks():
    def analyze_and_insert_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether it contains advice and an associated task and insert it into the database accordingly. Follow these steps:
                1. Check if any part of the USER_MESSAGE asks the agent to perform a task or solve a problem.
                If not, stop here.
                2. If the USER_MESSAGE asks the agent to perform a task or solve a problem, try to extract any advice from the USER_MESSAGE that may be useful for a similar but different task in the future. 
                3. If advice is found, extract just the task from the USER_MESSAGE. Don't solve it or include any advice.
                4. Summarize the extracted task very briefly and in general terms. Leave out details that might not appear in a similar problem.
                5. If a task and advice pair is identified, use the insert_memory tool, with the generalized task as the input and the advice as the output to insert the pair into the database.
            
            Take a deep breath and think step by step.
            """,
            expected_output=f"""The task and advice pair that was inserted into the database.""",
            agent=agent,
            async_execution=False
    )

    def analyze_and_insert_meaningful_info_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether it contains meaningful information and insert it into the database accordingly. Follow these steps:
                1. Check if the USER_MESSAGE contains information that could be committed to memory.
                2. If so, imagine that the user forgot this information in the USER_MESSAGE. Formulate a question that the user would ask to retrieve this information.
                3. Extract the information from the USER_MESSAGE that answers the question and should be committed to memory.
                4. If a question and answer pair is identified, use the insert_memory tool, with the question as the input and the answer as the output to insert the pair into the database.
            
            Take a deep breath, think step by step.
            """,
            expected_output=f"""The question and answer pair that was inserted into the database.""",
            agent=agent,
            async_execution=False
    )


    def retrieve_and_combine_relevant_memories_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether to retrieve memos from the database. Follow these steps:
                1. Use the entire USER_MESSAGE as the lookup key to retrieve relevant memos from the database utilizing the retrieve_memories tool, treating them as question-answer pairs.
                2. Check if any part of the USER_MESSAGE asks the agent to perform a task or solve a problem. 
                3. If so, extract just the task from the USER_MESSAGE. Don't solve it or include any advice.
                4. Summarize the extracted task very briefly and in general terms. Leave out details that might not appear in a similar problem.
                5. Use the generalized task as the lookup key to retrieve additional relevant memos from the database utilizing the retrieve_memories tool, treating them as task-advice pairs.
                6. Combine the memos retrieved using the entire USER_MESSAGE (question-answer pairs) and the memos retrieved using the generalized task (task-advice pairs) into a single list.
                Use the concatenate_memo_texts tool to do so.
                7. Provide the final combined memos as a neat bulleted list.
                8. Remove any duplicate memos from the list to avoid redundancy.
            """,
            expected_output=f"""The relevant memos retrieved from the database in a neat bulleted list with duplicates removed.""",
            agent=agent,
            async_execution=False
    )


  

