from langchain.agents import tool
from memo_store import MemoStore
from formatting_utils import colored
from typing import Dict, Optional, Union
from memo_store import MemoStore
from typing import Type

class MemoryTools():
  @tool
  def insert_memory(memo_store: Type[MemoStore], general_task: str, advice: str):
        """Store something from one user comment in the DB."""
        memo_store.add_input_output_pair(general_task, advice)
        memo_store._save_memos()


  def tools():
    return [
      MemoryTools.insert_memory
    ]
