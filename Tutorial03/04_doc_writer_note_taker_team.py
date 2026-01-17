import functools
import operator
import re
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# CONFIGURATION - FIXED: Use persistent directory
# ============================================================================

WORKING_DIRECTORY = Path("./output")
WORKING_DIRECTORY.mkdir(exist_ok=True)

# Initialize the Groq LLM
llm = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.1,
    model_kwargs={"top_p": 0.5, "seed": 1337}
)

# ============================================================================
# TOOLS
# ============================================================================

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    
    sorted_inserts = sorted(inserts.items())
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."
    
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)
    return f"Document edited and saved to {file_name}"

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    return f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"

# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(MessagesState):
    """State for the document writing team."""
    next: str

# ============================================================================
# SUPERVISOR NODE - IMPROVED PROMPTING
# ============================================================================

def make_supervisor_node(llm: ChatGroq, members: List[str]):
    """Create a supervisor node that routes work to team members using text parsing."""
    system_prompt = (
        "You are a supervisor managing a team of workers. "
        "Analyze the user request and conversation history to decide who works next."
        "\n\nYour team:"
        "\n- note_taker: Creates outlines and structured notes ONLY (cannot write full documents)"
        "\n- doc_writer: Writes complete documents, poems, articles to disk"
        "\n- chart_generator: Creates charts and visualizations with Python"
        "\n\nRouting rules:"
        "\n1. If task needs an outline/structure first → choose note_taker"
        "\n2. If outline exists or task needs full content written → choose doc_writer"
        "\n3. If task needs charts/graphs → choose chart_generator"
        "\n4. If all work is complete → respond FINISH"
        "\n\nRespond with ONLY ONE word: note_taker, doc_writer, chart_generator, or FINISH"
    )
    
    def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
        """An LLM-based router using text parsing."""
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]
        
        response = llm.invoke(messages)
        response_text = response.content.strip().upper()
        
        # Parse the response to extract the worker name
        goto = None
        for member in members:
            if member.upper() in response_text:
                goto = member
                break
        
        if goto is None or "FINISH" in response_text:
            goto = END
        
        print(f"Supervisor decision: {goto if goto != END else 'FINISH'}")
        
        return Command(goto=goto, update={"next": str(goto)})
    
    return supervisor_node

# ============================================================================
# AGENTS
# ============================================================================

doc_writer_agent = create_agent(
    llm,
    tools=[write_document, edit_document, read_document],
    system_prompt=(
        "You are a document writer. Write complete, polished documents. "
        "Use write_document to save content to disk. "
        "You write full content, not outlines. "
        "Always confirm the filename when done."
    ),
)

def doc_writing_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for document writing tasks."""
    result = doc_writer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ]
        },
        goto="supervisor",
    )

note_taking_agent = create_agent(
    llm,
    tools=[create_outline, read_document],
    system_prompt=(
        "You are a note taker. You ONLY create outlines and structured notes. "
        "Use create_outline to save structured outlines. "
        "You do NOT write full documents - that's the doc_writer's job."
    ),
)

def note_taking_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for note taking and outline creation."""
    result = note_taking_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ]
        },
        goto="supervisor",
    )

chart_generating_agent = create_agent(
    llm, 
    tools=[read_document, python_repl_tool],
    system_prompt=(
        "You are a chart generator. Create visualizations using matplotlib. "
        "Read documents with read_document, then use python_repl_tool to generate charts."
    ),
)

def chart_generating_node(state: State) -> Command[Literal["supervisor"]]:
    """Node for chart generation."""
    result = chart_generating_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content, name="chart_generator"
                )
            ]
        },
        goto="supervisor",
    )

# ============================================================================
# BUILD THE GRAPH
# ============================================================================

doc_writing_supervisor_node = make_supervisor_node(
    llm, ["doc_writer", "note_taker", "chart_generator"]
)

authoring_graph = StateGraph(State)
authoring_graph.add_node("supervisor", doc_writing_supervisor_node)
authoring_graph.add_node("doc_writer", doc_writing_node)
authoring_graph.add_node("note_taker", note_taking_node)
authoring_graph.add_node("chart_generator", chart_generating_node)

authoring_graph.add_edge(START, "supervisor")

chain = authoring_graph.compile()

# ============================================================================
# RUN THE GRAPH
# ============================================================================

if __name__ == "__main__":
    print(f"Working directory: {WORKING_DIRECTORY.absolute()}\n")
    
    for s in chain.stream(
        {
            "messages": [
                ("user", "Write an outline for a poem and then write the poem to disk.")
            ]
        },
        {"recursion_limit": 100},
    ):
        if "__end__" not in s:
            print(s)
            print("---")
    
    print("\n\nFiles created:")
    for f in WORKING_DIRECTORY.rglob("*"):
        if f.is_file():
            print(f"  📄 {f.name}")
            content = f.read_text()
            print(f"     Preview: {content[:80]}...")
            print(f"     Size: {len(content)} chars\n")
