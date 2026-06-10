"""LangGraph ReAct agent graph with max-iterations guard."""

import os
import re

import litellm
from langchain_core.messages import AIMessage
from langchain_litellm import ChatLiteLLM
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.agent.tools import TOOLS


def _get_llm() -> ChatLiteLLM:
    """Return a configured ChatLiteLLM bound with tools, routing through the CMU AI Gateway."""
    return ChatLiteLLM(
        model="openai/wine-gemini-3-flash-preview",
        api_key=os.getenv("LITELLM_API_KEY"),
        api_base=os.getenv("LITELLM_API_BASE"),
    ).bind_tools(TOOLS)


_raw_tool_node = ToolNode(TOOLS)


def agent_node(state: AgentState) -> dict:
    """Call the LLM with current messages and increment step counter."""
    llm = _get_llm()
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


def tools_node(state: AgentState) -> dict:
    """Execute tool calls and extract source references from results."""
    result = _raw_tool_node.invoke(state)
    current_sources = list(state.get("sources", []))

    for msg in result["messages"]:
        if not hasattr(msg, "name"):
            continue
        if msg.name == "search_docs":
            found = re.findall(r"\[Source: ([^\]]+)\]", str(msg.content))
            current_sources.extend(found)
        elif msg.name == "web_search":
            if "web_search" not in current_sources:
                current_sources.append("web_search")

    return {
        "messages": result["messages"],
        "sources": list(dict.fromkeys(current_sources)),
    }


def max_iter_node(state: AgentState) -> dict:
    """Inject a max-iterations message before terminating."""
    return {
        "messages": [
            AIMessage(content="Max iterations reached. Please try a more specific query.")
        ]
    }


def should_continue(state: AgentState) -> str:
    """Route to tools, END, or max_iter based on step count and tool calls."""
    if state.get("step_count", 0) >= 8:
        return "max_iter"
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_graph():
    """Compile and return the ReAct agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("max_iter", max_iter_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END, "max_iter": "max_iter"},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("max_iter", END)

    return workflow.compile()
