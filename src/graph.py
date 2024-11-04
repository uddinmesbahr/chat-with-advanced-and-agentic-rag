from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from .state import AgentState
from .nodes import Nodes

from .crew.crew import EmailFilterCrew

class WorkflowGraph:
    def __init__(self, agent_state: AgentState):
        self.workflow = StateGraph(agent_state)
        self.nodes_instance = Nodes()  # Create an instance of Nodes
        self.workflow.add_node("router", self.nodes_instance.router)
        self.workflow.add_node("web_search", self.nodes_instance.web_search)
        self.workflow.add_node("vectorstore_retrieve", self.nodes_instance.vectorstore_retrieve)
        self.workflow.add_node("cypher_retriever", self.nodes_instance.cypher_retriever)
        self.workflow.add_node("cypher_translating", self.nodes_instance.cypher_translating)
        self.workflow.add_node("retrieve_grader", self.nodes_instance.retrieve_grader)
        self.workflow.add_node("generate", self.nodes_instance.generate)
        self.workflow.add_node("mutiple_question_generators", self.nodes_instance.multiple_question_generators)
        self.workflow.add_node("reciprocal_rank_fusion", self.nodes_instance.reciprocal_rank_fusion)
        self.workflow.add_node("hallucination_grader", self.nodes_instance.hallucination_grader)
        self.workflow.add_node("grader_crew", self.email_filter_crew.grader_crew)
        self.workflow.add_node("hallucination_crew", self.email_filter_crew.hallucination_crew)
        self.workflow.add_node("generation_crew", self.email_filter_crew.generation_crew)

        # Define edges
        self.workflow.add_edge(START, "router")
        self.workflow.add_conditional_edges(
            "router",
            self.nodes_instance.route_decision,
            {
                "web_search": "web_search",
                "vectorstore": "vectorstore_retrieve",
                "cypher db": "cypher_translating",
            },
        )
        self.workflow.add_edge("web_search", "generate")
        self.workflow.add_edge("cypher_translating", "cypher_retriever")
        self.workflow.add_edge("vectorstore_retrieve", "retrieve_grader")
        self.workflow.add_conditional_edges(
            "retrieve_grader",
            self.nodes_instance.decide_to_generate,
            {
                "mutiple_question_generators": "mutiple_question_generators",
                "generate": "generate",
            },
        )
        self.workflow.add_edge("mutiple_question_generators", "vectorstore_retrieve")
        self.workflow.add_edge("vectorstore_retrieve", "reciprocal_rank_fusion")
        
        self.workflow.add_edge("reciprocal_rank_fusion", "generate")
        self.workflow.add_edge("generate", "final_grader")
        self.app = self.workflow.compile()

