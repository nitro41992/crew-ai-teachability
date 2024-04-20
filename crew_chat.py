from groq import Groq
from openai import OpenAI
import os
from crewai import Crew
from tasks import MemoryTasks
from agents import LTMAgents
from dotenv import load_dotenv
load_dotenv()

# model = 'llama3-70b-8192' 
# api_key = os.environ.get("GROQ_API_KEY")
# os.environ['OPENAI_API_BASE']='https://api.groq.com/openai/v1'
# os.environ['OPENAI_MODEL_NAME']=model  # Adjust based on available model
# os.environ['OPENAI_API_KEY']=api_key
# client = Groq(api_key=api_key)

model = 'gpt-3.5-turbo-0125' 
os.environ['OPENAI_MODEL_NAME']=model # Adjust based on available model
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# create tasks
tasks = MemoryTasks()
agents = LTMAgents()

# create agents
context_agent = agents.context_agent()
retrieval_agent = agents.retrieval_agent()


def run_crew(user_message):
    
    analyze_and_insert_task = tasks.analyze_and_insert_task(context_agent, user_message)
    analyze_and_insert_meaningful_info_task = tasks.analyze_and_insert_meaningful_info_task(context_agent, user_message)
    retrieve_and_combine_relevant_question_answers_task = tasks.retrieve_and_combine_relevant_question_answers_task(retrieval_agent, user_message)
    retrieve_and_combine_relevant_task_advice_task = tasks.retrieve_and_combine_relevant_task_advice_task(retrieval_agent, user_message)

    # Providing necessary context to the subsequent task
    retrieve_and_combine_relevant_task_advice_task.context = [retrieve_and_combine_relevant_question_answers_task]

    crew = Crew(
        agents=[
            context_agent,
            retrieval_agent
        ],
        tasks=[
            analyze_and_insert_task,
            analyze_and_insert_meaningful_info_task,
            retrieve_and_combine_relevant_question_answers_task,
            retrieve_and_combine_relevant_task_advice_task
        ]
    )
    return(crew)

def chat():
    messages = [
        {
            "role": "system",
            "content":"""
            You are a helpful assistant. You specialize in providing nutrition advice and meal plans. You are very cognizant of the customer's
            preferences and restrictions when providing nutrition advice and meal plans. 
            Additional context and personalized information will be provided to you. Use it to respond to the best of your ability to the user.
            """
        }
    ]

    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'quit':
            print("Chat terminated by the user.")
            break
        
        crew_analysis = run_crew(user_input)
        results = crew_analysis.kickoff()
        messages.append({"role": "user", "content": user_input + results})
        # messages.append({"role": "user", "content": user_input})

        # print(messages)


        stream = client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        full_response = ""
        print("Assistant: ", end="")
        for chunk in stream:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

        messages.append({"role": "assistant", "content": full_response})
        # print(messages)


if __name__ == "__main__":
    print("Welcome to the Chat Interface!")
    print("You can start chatting with the assistant now.")
    print("Type 'quit' to exit the chat.")
    print()

    try:
        chat()
    except KeyboardInterrupt:
        print("\nChat terminated by the user.")