from typing import Annotated, Sequence, TypedDict, List
from langchain_core.messages import BaseMessage
# from langgraph.graph.message import add_messages



class AgentState(TypedDict):
    
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """

    question: str
    generation: str
    documents: List[str]