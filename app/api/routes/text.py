import os
from typing import Dict
from fastapi import APIRouter

from google.genai import Client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage

router = APIRouter()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

client = Client(
  api_key=GOOGLE_API_KEY,
  http_options= {'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"

memory: MemorySaver = MemorySaver()

config = {"configurable": {"thread_id": "1"}, "generation_config": {"response_modalities": ["TEXT"]}}

class State(MessagesState):
  pass

async def call_module(state: State) -> Dict[str, object]:
    messages = state.get("messages", [])

    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send(input=f"{history_str}\nUser: {messages[-1].content}", end_of_turn=True)
        turn = session.receive()
        response_text = ""
        async for chunk in turn:
            if chunk.text is not None:
                response_text += chunk.text

    updated_messages = messages + [AIMessage(content=response_text)]

    return {
        "response": response_text,
        "messages": updated_messages
    }

workflow:StateGraph = StateGraph(State)

workflow.add_node("agent", call_module)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile(checkpointer=memory)

@router.get("/")
async def run(message:str):
    event = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, config)
    return event