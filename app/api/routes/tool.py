import os
from fastapi import APIRouter
from google.genai import Client
from typing import Literal, Union
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

router = APIRouter()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = Client(
  api_key=GOOGLE_API_KEY,
  http_options= {'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"

memory: MemorySaver = MemorySaver()

google_search_tool = Tool(
    google_search = GoogleSearch()
)

config = {
    "configurable": {"thread_id": "1"},
    }

# State

class State(MessagesState):
    summary: str


# Summarization

def summarize_conversation(state: State):
    """
    Summarizes the conversation if the number of messages exceeds 6 messages.
    
    Args:
        state (State): The current conversation state.
        model (object): The model to use for summarization.

    Returns:
        Dict[str, object]: A dictionary containing updated messages.
    """
   
    # Get any existing summary
    summary = state.get("summary", "")

    # Create summarization prompt based on whether there is an existing summary
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    # Add the summarization prompt to the conversation history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = client.models.generate_content(
        model=MODEL,
        contents=messages,
        config=GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )
    )

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]
    
    return {"summary": response.text, "messages": delete_messages}

# Conditional Function

def select_next_node(state: State) -> Union[Literal["summarize"], str]:

    messages = state["messages"]
    last_message = messages[-1]

    # If there are more than six messages, route to "summarize_conversation"
    if len(messages) > 6:
        return "summarize"
    
    # Otherwise, route to "final" or end
    return END

# Invoke Messages

def call_model(state: State):

    # Ensure state contains 'messages'
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")
    
    # Initialize messages from the state
    messages = state["messages"]
    
    # Check if a summary exists and prepend it as a system message if present
    summary = state.get("summary", "")
    if summary:
        system_message = f"Summary of conversation earlier: {summary}"
        messages = [SystemMessage(content=system_message)] + messages

    # Safely invoke the model
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=messages,
            config=GenerateContentConfig(
                response_modalities=["TEXT"],
            )
        )

    except Exception as e:
        raise RuntimeError(f"Error invoking the model: {e}")

    # Append the response to messages
    messages.append(response.text)

    # Return the updated state with messages
    return {"messages": messages[-1]}

# Build Graph

builder = StateGraph(State)

builder.add_node("agent", call_model)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "agent")
builder.add_edge("summarize", END)

builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", END: END},
)

graph = builder.compile(checkpointer=memory)

@router.get("/")
async def run(message: str):
    # Invoke the state graph with the initial messages
    event = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, config)
    return event