from langchain.agents import tool
from memo_store import MemoStore

memo_store = MemoStore(verbosity=0, reset=False, path_to_db_dir="./tmp/teachable_agent_db")

class MemoryTools():
    @tool
    def insert_memory(property_type: str, property_value: str):
        """Store the property type and property as a memory from one user message into the memory store."""
        memo_store.add_input_output_pair(property_type, property_value)
        memo_store._save_memos()


    def tools():
        return [
            MemoryTools.insert_memory
        ]
