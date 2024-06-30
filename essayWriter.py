from dotenv import load_dotenv
import os

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage


# Load the environment variables
load_dotenv()

# Checkpoint of the state graph
memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    task: str # writing essay
    plan: str # how to write essay
    draft: str # draft generated
    critique: str # critique of draft generated
    content: List[str] # content that tavilly found
    revision_number: int # number of times reviewed/Loop
    max_revisions: int # max number of loops allowed




