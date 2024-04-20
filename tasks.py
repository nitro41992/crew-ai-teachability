from crewai import Task

class MemoryTasks():
    def analyze_and_insert_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether it contains advice and an associated task and insert it into the database accordingly. Follow these steps:
                1. Determine if any part of the USER_MESSAGE asks the agent to perform a task or solve a problem.
                2. If the USER_MESSAGE asks the agent to perform a task or solve a problem, try to extract any advice from the USER_MESSAGE that may be useful for a similar but different task in the future. 
                If not, don't insert anything respond with "No tasks found.", and do not proceed with the rest of the steps.
                3. If advice is found, extract just the task from the USER_MESSAGE.
                4. Summarize the extracted task very briefly and in general terms. Leave out details that might not appear in a similar problem.
                5. If a task and advice pair is identified, use the insert_memory tool, with the generalized task as the input and the advice as the output to insert the pair into the database.
                6. If nothing is found, respond with "No task and advice pairs found."
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
                1. Determine if the USER_MESSAGE contains information that could be committed to memory.
                2. If so, imagine that the user forgot this information in the USER_MESSAGE. Formulate a question that the user would ask to retrieve this information.
                3. Extract the information from the USER_MESSAGE that answers the question and should be committed to memory.
                4. If a question and answer pair is identified, use the insert_memory tool, with the question as the input and the answer as the output to insert the pair into the database.
                5. If nothing is found, respond with "No question and answer pairs found."
            Take a deep breath, think step by step.
            """,
            expected_output=f"""The question and answer pair that was inserted into the database.""",
            agent=agent,
            async_execution=False
    )


    def retrieve_and_combine_relevant_question_answers_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether to retrieve memos from the database. Follow these steps:
                1. Use the entire USER_MESSAGE as the lookup key to retrieve relevant memos from the database utilizing the retrieve_memories tool, treating them as question-answer pairs.
                2. Determine if any part of the USER_MESSAGE asks the agent to perform a task or solve a problem. 
                3. If so, extract just the task from the USER_MESSAGE. Don't solve it or include any advice.
                4. Combine the memos retrieved using the entire USER_MESSAGE (question-answer pairs) into a single list. Use the concatenate_memo_texts tool to do so.
                5. Provide the final combined memos as a neat bulleted list.
                6. Remove any duplicate memos from the list to avoid redundancy.
            """,
            expected_output=f"""The relevant memos retrieved from the database in a neat bulleted list with duplicates removed.""",
            agent=agent,
            async_execution=False
    )

    def retrieve_and_combine_relevant_task_advice_task(self, agent, user_comment):
        return Task(
            description=f"""
            Here is the USER_MESSAGE: {user_comment}
            Analyze the given USER_MESSAGE and decide whether to retrieve memos from the database. Follow these steps:
                1. Use the entire USER_MESSAGE as the lookup key to retrieve relevant memos from the database utilizing the retrieve_memories tool, treating them as task-advice pairs.
                2. Determine if any part of the USER_MESSAGE asks the agent to perform a task or solve a problem. 
                3. If so, extract just the task from the USER_MESSAGE. Don't solve it or include any advice.
                4. Combine the memos retrieved using the entire USER_MESSAGE (task-advice pairs) into a single list. Use the concatenate_memo_texts tool to do so.
                5. Provide the final combined memos as a neat bulleted list. Get the memos provided from the retrieve_and_combine_relevant_question_answers_task and add them to the list.
                6. Remove any duplicate memos from the list to avoid redundancy.
            """,
            expected_output=f"""The relevant memos retrieved from the database in a neat bulleted list with duplicates removed.""",
            agent=agent,
            async_execution=False
    )



  

