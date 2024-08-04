from dotenv import load_dotenv
import os
import time

from langgraph.graph import StateGraph, END
from typing import TypedDict, List
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage

from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_core.pydantic_v1 import BaseModel

from templates import PLAN_PROMPT, WRITER_PROMPT, REFLECTION_PROMPT, RESEARCH_CRITIQUE_PROMPT, RESEARCH_PLAN_PROMPT

from openai import RateLimitError  # Import RateLimitError specifically

# Load the environment variables
load_dotenv()

# Checkpoint of the state graph
memory = SqliteSaver.from_conn_string(":memory:")

class AgentState(TypedDict):
    """
    State of the agent
    Each variable represents a node in the graph
    """
    task: str # writing essay
    plan: str # how to write essay
    draft: str # draft generated
    critique: str # critique of draft generated
    content: List[str] # content that tavilly found
    revision_number: int # number of times reviewed/Loop
    max_revisions: int # max number of loops allowed

class Queries(BaseModel):
    """Queries to be used for research"""
    queries: List[str]

def call_openai_api_with_backoff(messages, retries=5):
    for i in range(retries):
        try:
            response = model.invoke(messages)
            return response
        except RateLimitError as e:
            print(f"Rate limit exceeded. Retrying in {2 ** i} seconds...")
            time.sleep(2 ** i)
        except Exception as e:
            print(f"An error occurred: {e}")
            break
    return None

def plan_node(state: AgentState):
    messages = [
        SystemMessage(content=PLAN_PROMPT), 
        HumanMessage(content=state['task'])
    ]
    response = call_openai_api_with_backoff(messages)
    if response:
        return {"plan": response.content}
    return {"plan": ""}

def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_PLAN_PROMPT),
        HumanMessage(content=state['task'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}")
    messages = [
        SystemMessage(
            content=WRITER_PROMPT.format(content=content)
        ),
        user_message
        ]
    response = call_openai_api_with_backoff(messages)
    if response:
        return {
            "draft": response.content, 
            "revision_number": state.get("revision_number", 1) + 1
        }
    return {
        "draft": "", 
        "revision_number": state.get("revision_number", 1) + 1
    }

def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT), 
        HumanMessage(content=state['draft'])
    ]
    response = call_openai_api_with_backoff(messages)
    if response:
        return {"critique": response.content}
    return {"critique": ""}

def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke([
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
        HumanMessage(content=state['critique'])
    ])
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}

def should_continue(state):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"

# This is a wrapper, so other models can be used in place here. 
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

builder = StateGraph(AgentState)
builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", 
    should_continue, 
    {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")

builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)

thread = {"configurable": {"thread_id": "1"}}
for s in graph.stream({
    'task': "Write about Donald Trump's presidency from 2016 to 2020. Discuss his policies and their impact on the United States.",
    "max_revisions": 2,
    "revision_number": 1,
}, thread):
    print(s)