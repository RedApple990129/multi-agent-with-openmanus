from langgraph.graph import StateGraph, START
from src.graph.types import State # Import State class
from src.agents.nodes import ( # Import agent nodes
    coordinator_node,
    planner_node,
    supervisor_node,
    researcher_node,
    coder_node,
    browser_node,
    reporter_node,
)

def build_graph():
    """Build and return the agent workflow graph."""
    builder = StateGraph(State)

    # Define nodes
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("coder", coder_node)
    builder.add_node("browser", browser_node)
    builder.add_node("reporter", reporter_node)

    # Define edges
    builder.add_edge(START, "coordinator")
    builder.add_edge("coordinator", "planner") # Coordinator -> Planner
    builder.add_edge("planner", "supervisor") # Planner -> Supervisor
    builder.add_edge("supervisor", "researcher", condition=lambda state: state['next'] == "researcher") # Supervisor -> Researcher if next agent is researcher
    builder.add_edge("supervisor", "coder", condition=lambda state: state['next'] == "coder") # Supervisor -> Coder if next agent is coder
    builder.add_edge("supervisor", "browser", condition=lambda state: state['next'] == "browser") # Supervisor -> Browser if next agent is browser
    builder.add_edge("supervisor", "reporter", condition=lambda state: state['next'] == "reporter") # Supervisor -> Reporter if next agent is reporter
    builder.add_edge("supervisor", "__end__", condition=lambda state: state['next'] == "__end__") # Supervisor -> END if next agent is FINISH
    builder.add_edge("researcher", "supervisor") # Researcher -> Supervisor
    builder.add_edge("coder", "supervisor") # Coder -> Supervisor
    builder.add_edge("browser", "supervisor") # Browser -> Supervisor
    builder.add_edge("reporter", "supervisor") # Reporter -> Supervisor

    builder.set_entry_point("coordinator")
    builder.set_conditional_edge("supervisor", supervisor_node) # Conditional edge for supervisor node
    builder.add_end_point("__end__")

    return builder.compile()