import os
import wave
import logging
import contextlib
from IPython.display import Audio

from google.genai import Client
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, START, END, StateGraph
from fastapi import APIRouter
from fastapi.responses import FileResponse
from dotenv import load_dotenv

load_dotenv()  
router = APIRouter()

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

client = Client(
  http_options= {'api_version': 'v1alpha'}
)

MODEL: str = "gemini-2.0-flash-exp"
FILE_NAME = 'audio.wav'
config={"configurable": {"thread_id": "1"}, "generation_config": {"response_modalities": ["AUDIO"]}}

logger = logging.getLogger('Live')
logger.setLevel('INFO')

memory: MemorySaver = MemorySaver()

class State(MessagesState):
  pass

@contextlib.contextmanager
def wave_file(filename, channels=1, rate=24000, sample_width=2):
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        yield wf
        
async def call_module(state: State, user_msg: str):

  state["messages"].append({"role": "user", "content": user_msg})

  convo_history = "\n".join([msg["content"] for msg in state["messages"] if msg["content"]])

  async with client.aio.live.connect(model=MODEL, config=config) as session:
    with wave_file(FILE_NAME) as wav:
      print("> ", user_msg, "\n")
      await session.send(convo_history, end_of_turn=True)

      receive = session.receive()

      # Track if this is the first response to display metadata
      first = True
      async for response in receive:
        if response.data is not None:
          model_turn = response.server_content.model_turn
          if first:
            print(model_turn.parts[0].inline_data.mime_type)
            first = False
          print('.', end='.')
          wav.writeframes(response.data)

      response_text = ""
      async for chunk in receive:
          if chunk.text is not None:
              response_text += chunk.text
              print(chunk.text, end="")

      print(response_text)

      # Display the audio file in a Jupyter/Colab notebook
      return Audio(FILE_NAME, autoplay=True), response_text
  

workflow:StateGraph = StateGraph(State)

workflow.add_node("agent", call_module)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

graph = workflow.compile(checkpointer=memory)

@router.get("/")
async def run(message: str):
    # Initialize state
    state = State(messages=[])
    
    # Call the module to process user input and generate audio
    audio, response_text = await call_module(state, message)
    
    # Return the audio file as a response
    return FileResponse(
        FILE_NAME,
        media_type="audio/wav",
        filename="output_audio.wav"
    )
