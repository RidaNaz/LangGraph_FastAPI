import os
from typing import Literal
from langchain import hub
from fastapi import APIRouter
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.prebuilt import tools_condition
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

router = APIRouter()

load_dotenv()
PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY=os.environ.get('GOOGLE_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"]= GOOGLE_API_KEY

# LLM
model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(temperature=0, model = "gemini-pro", streaming=True)

memory: MemorySaver = MemorySaver()

config = {"configurable": {"thread_id": "4"}}

class State(MessagesState):
  summary: str
  
#Download the Embeddings from Hugging Face
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

index_name = "medical-bot"
embeddings = download_hugging_face_embeddings()

# Load Existing index 
# Embed each chunk and upsert the embeddings into your Pinecone index.
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know. keep the answer concise.",
)

tools= [retriever_tool]

### Edges

def grade_documents(state:State) -> Literal["generate", "rewrite"]:
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (messages): The current state

    Returns:
        str: A decision for whether the documents are relevant or not
    """

    print("---CHECK RELEVANCE---")

    # Data model
    class grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(temperature=0, model = "gemini-pro", streaming=True)

    # LLM with tool and validation
    llm_with_tool = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]

    question = messages[-3].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"
    
### Nodes

def agent(state: State):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(temperature=0, model = "gemini-pro", streaming=True)
    
    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def rewrite(state: State):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    question = messages[-1].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Grader
    model: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(temperature=0, model = "gemini-pro", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[-3].content
    last_message = messages[-1]

    docs = last_message.content

    # Prompt
    prompt = hub.pull("rlm/rag-prompt")

    # LLM
    llm = ChatGoogleGenerativeAI(temperature=0, model = "gemini-pro", streaming=True)
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# Define a new graph
workflow = StateGraph(State)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent

retrieve = ToolNode([retriever_tool])

workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node("generate", generate)
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)

workflow.add_edge("rewrite", "agent")
workflow.add_edge("generate", END)

# Compile
graph = workflow.compile(checkpointer=memory)

@router.get("/")
async def run(message: str):
    # Invoke the state graph with the initial messages
    event = await graph.ainvoke({"messages": [HumanMessage(content=message)]}, config)
    return event

