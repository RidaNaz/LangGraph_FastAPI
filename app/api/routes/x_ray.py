import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from google.genai import Client
from typing import Literal, Union, Dict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
import torch
import io
from PIL import Image

router = APIRouter()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

client = Client(
    api_key=GOOGLE_API_KEY,
    http_options={'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"
memory: MemorySaver = MemorySaver()

config = {
    "configurable": {"thread_id": "1"},
    "generation_config": {"response_modalities": ["TEXT"]},
}

# Load the model and feature extractor
model_name = "lxyuan/vit-xray-pneumonia-classification"
model = AutoModelForImageClassification.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# State Class
class State(MessagesState):
    summary: str

# Image Processing Logic (new node)
async def image_processing(state: State) -> Dict[str, object]:
    try:
        # Extract image from state (assuming state contains an uploaded image file)
        image_data = state.get("image_data")
        if not image_data:
            raise ValueError("No image data found in the state.")

        # Read the uploaded image file
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        # Add the prediction to the state as a message
        state["prediction"] = predicted_class_idx
        state["messages"].append(HumanMessage(content=f"Predicted class: {predicted_class_idx}"))

        return {"prediction": predicted_class_idx, "messages": state["messages"]}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

# Summarization Logic (same as before)
async def summarize_conversation(state: State) -> Dict[str, object]:
    summary = state.get("summary", [])
    if summary:
        summary_message = (
            f"This is the summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"

    messages = [SystemMessage(content=summary_message)] + state["messages"]
    history_str = "\n".join([f"{m.type}: {m.content}" for m in messages])
    input_text = f"{history_str}\nUser: {summary_message}"

    async with client.aio.live.connect(model=MODEL, config=config) as session:
        await session.send(input=input_text, end_of_turn=True)
        turn = session.receive()
        response_text = ""
        async for chunk in turn:
            if chunk.text is not None:
                response_text += chunk.text

    state["summary"] = response_text
    delete_messages = [RemoveMessage(id=getattr(m, "id", None)) for m in state["messages"][:-2]]
    return {"summary": response_text, "messages": delete_messages}

# Conditional Function (same as before)
def select_next_node(state: State) -> Union[Literal["summarize"], str]:
    messages = state["messages"]
    if len(messages) > 6:
        return "summarize"
    return END

# Model Invocation Logic (updated)
async def call_model(state: State):
    if "messages" not in state:
        raise ValueError("State must contain a 'messages' key.")
    
    # Retrieve the prediction result from the state
    predicted_class_idx = state.get("prediction")
    if predicted_class_idx is None:
        raise ValueError("Prediction not found in the state.")
    
    # Generate a response based on the prediction
    if predicted_class_idx == 1:
        response_text = "The X-ray result indicates potential pneumonia. Please consult with a healthcare provider for further examination."
    elif predicted_class_idx == 0:
        response_text = "The X-ray result does not indicate pneumonia. However, you should continue monitoring your health and consult a doctor if symptoms persist."
    else:
        response_text = f"The prediction returned an unexpected result: {predicted_class_idx}. Please consult a healthcare provider for further analysis."

    # Add the response to the conversation
    response_message = AIMessage(content=response_text)
    state["messages"].append(response_message)

    return {"messages": state["messages"], "last_message": response_message}


# Build the Graph
builder = StateGraph(State)
builder.add_node("image_processing", image_processing)
builder.add_node("agent", call_model)
builder.add_node("summarize", summarize_conversation)

builder.add_edge(START, "image_processing")
builder.add_edge("image_processing", "agent")
builder.add_edge("summarize", END)
builder.add_conditional_edges(
    "agent",
    select_next_node,
    {"summarize": "summarize", END: END},
)
graph = builder.compile(checkpointer=memory)

@router.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        image_data = await file.read()

        # Create the state with the image data
        state = State(messages=[], image_data=image_data)

        # Call the image processing node to get the prediction
        result = await image_processing(state)

        # Pass the result to the agent node for further processing
        agent_result = await call_model(state)

        # Return the result from the agent
        return {
            "prediction": result["prediction"],
            "agent_response": agent_result["last_message"].content
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
