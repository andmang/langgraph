import operator
from typing import Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage, HumanMessage, AnyMessage
from langgraph.graph import StateGraph, START, END

# Import tools from our local file
from tools import tools, tools_by_name

# --- Step 1: Define Model (Groq) ---

# We use init_chat_model with the Groq provider.
model = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0
)

# Augment the LLM with tools
model_with_tools = model.bind_tools(tools)


# --- Step 2: Define State ---

class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


# --- Step 3: Define Nodes ---

def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }


def tool_node(state: MessagesState):
    """Performs the tool call"""
    result = []
    last_message = state["messages"][-1]
    
    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        # Invoke the tool
        observation = tool.invoke(tool_call["args"])
        # Create a ToolMessage with the result
        result.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
    return {"messages": result}


# --- Step 4: Define Logic (Conditional Edges) ---

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# --- Step 5: Build Agent Graph ---

agent_builder = StateGraph(MessagesState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


# --- Step 6: Execution ---

if __name__ == "__main__":
    print("\n--- Running Agent (Groq) ---\n")
    
    # Example Query
    user_query = "Add 3 and 4, then multiply the result by 10."
    print(f"User: {user_query}")
    
    user_input = [HumanMessage(content=user_query)]
    
    # Run the graph
    output = agent.invoke({"messages": user_input})
    
    print("\n--- Conversation History ---\n")
    for m in output["messages"]:
        m.pretty_print()