from langchain.agents import tool
from memo_store import MemoStore

memo_store = MemoStore(verbosity=0, reset=True, path_to_db_dir="./tmp/teachable_agent_db")
max_number_of_retrieved_results = 10
max_threshold = 1.5

class MemoryTools():
    @tool
    def insert_memory(input_text: str, output_text: str):
        """Inserts the input_text and output_text extracted from the message into the database."""
        memo_store.add_input_output_pair(input_text, output_text)
        memo_store._save_memos()

    @tool
    def retrieve_memories(input_text: str) -> list:
        """Returns semantically related memos from the DB."""
        try:
            memo_list = memo_store.get_related_memos(
                input_text,
                n_results=max_number_of_retrieved_results,
                threshold=max_threshold
            )
            if len(memo_list) == 0:
                nearest_memo = memo_store.get_nearest_memo(input_text)
                if nearest_memo is not None:
                    memo_list = [nearest_memo]
                else:
                    return []  # Return an empty list if no related memos are found
            
            # Create a list of just the memo output_text strings.
            output_texts = [memo[1] for memo in memo_list]
            return output_texts
        
        except IndexError:
            print("Error: List index out of range.")
            # Handle the specific case that caused the index out of range error
            # You can add debugging statements or return an appropriate value
            return []  # Return an empty list as a fallback
        
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            # Handle any other unexpected errors
            return []  # Return an empty list as a fallback
    
    @tool
    def concatenate_memo_texts(memo_list: list) -> str:
        """Concatenates the memo texts into a single string for inclusion in the chat context."""
        memo_texts = ""
        if len(memo_list) > 0:
            info = "\n# Memories that might help\n"
            for memo in memo_list:
                info = info + "- " + memo + "\n"
            memo_texts = memo_texts + "\n" + info
        return memo_texts


    def tools():
        return [
            MemoryTools.insert_memory,
            MemoryTools.retrieve_memories,
            MemoryTools.concatenate_memo_texts,
        ]
