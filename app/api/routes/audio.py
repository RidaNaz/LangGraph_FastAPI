import os
import wave
import logging
import contextlib
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import FileResponse
from google.genai import Client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, START, END
from pydantic import BaseModel
import pyttsx3
import speech_recognition as sr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

router = APIRouter()

# Initialize pyttsx3 for text-to-speech
engine = pyttsx3.init()

# Set up Google Gemini API client
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
client = Client(http_options={'api_version': 'v1alpha'})
MODEL = "gemini-2.0-flash-exp"
generation_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    "response_modalities": ["TEXT"]
}

FILE_NAME = 'audio.wav'

# Set up memory for LangGraph
memory = MemorySaver()
logger = logging.getLogger('Live')
logger.setLevel('INFO')

# State class for LangGraph
class State(MessagesState):
    pass

# Context manager to handle WAV file creation
@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf

# LangGraph module call
async def call_module(state: State, user_msg: str):
    state["messages"].append({"role": "user", "content": user_msg})
    convo_history = "\n".join([msg["content"] for msg in state["messages"] if msg["content"]])

    async with client.aio.live.connect(model=MODEL, config={"generation_config": generation_config}) as session:
        with wave_file(FILE_NAME) as wav:
            await session.send(convo_history, end_of_turn=True)
            receive = session.receive()

            # Process response and write to wave file
            first = True
            async for response in receive:
                if response.data is not None:
                    model_turn = response.server_content.model_turn
                    if first:
                        print(model_turn.parts[0].inline_data.mime_type)
                        first = False
                    wav.writeframes(response.data)

            response_text = ""
            async for chunk in receive:
                if chunk.text is not None:
                    response_text += chunk.text
                    print(chunk.text, end="")

            return FileResponse(FILE_NAME, media_type="audio/wav", filename="output_audio.wav"), response_text
        
        
workflow:StateGraph = StateGraph(State)

workflow.add_node("agent", call_module)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile(checkpointer=memory)

# FastAPI endpoint to receive user message and return generated audio
@router.post("/process_speech/")
async def process_speech(message: str, background_tasks: BackgroundTasks):
    state = State(messages=[])

    background_tasks.add_task(handle_gemini_request, state, message)
    return {"message": "Processing your request. Audio will be ready shortly."}

async def handle_gemini_request(state: State, user_input: str):
    # Call module to generate audio
    audio_response, _ = await call_module(state, user_input)

    # Play audio response
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1.0)
    engine.setProperty('voice', 'english-us')
    engine.say("Here is your response:")
    engine.runAndWait()

    return audio_response

# FastAPI root route
@router.get("/")
async def root():
    return {"message": "Welcome to LangGraph + FastAPI speech processing service!"}

# Mount the router to the FastAPI app
router.include_router(router)