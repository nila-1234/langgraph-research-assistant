"""Agent state definition for the ReAct research agent."""

from typing import Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State passed between nodes in the LangGraph agent."""

    messages: Annotated[list[BaseMessage], add_messages]
    sources: list[str]
    step_count: int
