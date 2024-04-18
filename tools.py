from langchain.agents import tool
from memo_store import MemoStore

memo_store = MemoStore(verbosity=0, reset=True, path_to_db_dir="./tmp/teachable_agent_db")
max_number_of_retrieved_results = 10
max_threshold = 3

class MemoryTools():
    @tool
    def insert_memory(property_type: str, property_value: str):
        """Store the property type and property as a memory from one user message into the memory store."""
        memo_store.add_input_output_pair(property_type, property_value)
        memo_store._save_memos()

    @tool
    def retrieve_memories(input_text: str) -> list:
        """Returns semantically related memos from the DB."""
        memo_list = memo_store.get_related_memos(
            input_text, n_results=max_number_of_retrieved_results, threshold=max_threshold
        )

        if len(memo_list) == 0:
            memo_store.get_nearest_memo(input_text)
            print()  # Print a blank line. The memo details were printed by get_nearest_memo().

        # Create a list of just the memo output_text strings.
        memo_list = [memo[1] for memo in memo_list]
        return memo_list
    
    @tool
    def concatenate_memo_texts(self, memo_list: list) -> str:
        """Concatenates the memo texts into a single string for inclusion in the chat context."""
        memo_texts = ""
        if len(memo_list) > 0:
            info = "\n# Memories that might help\n"
            for memo in memo_list:
                info = info + "- " + memo + "\n"
            if self.verbosity >= 1:
                print("\nMEMOS APPENDED TO LAST MESSAGE...\n" + info + "\n", "light_yellow")
            memo_texts = memo_texts + "\n" + info
        return memo_texts


    def tools():
        return [
            MemoryTools.insert_memory,
            MemoryTools.retrieve_memories,
            MemoryTools.concatenate_memo_texts,
        ]
