from textwrap import dedent
from crewai import Agent
import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Load LLMs from environment variables
### use different llm to optimize the price and performance
router_llm = os.getenv('ROUTER_LLM')
grader_llm = os.getenv('GRADER_LLM')
answer_generator_llm = os.getenv('ANSWER_GENERATOR_LLM')
question_generators_llm = os.getenv('GENERATE_NEW_QUERY_LLM')
hallucination_llm = os.getenv('HALLUCINATION_LLM')
answer_review_llm = os.getenv('ANSWER_GRADER_LLM')  # Fixed typo
cypher_translator_llm = os.getenv('CYPHER_TRANSLATOR_LLM')


class RAG_AGENTS():
    """Class for managing Retrieval-Augmented Generation agents."""

    def __init__(self, router_llm, grader_llm, answer_generator_llm,
                 question_generators_llm, hallucination_llm, answer_review_llm, 
                 cypher_translator_llm):
        self.router_llm = router_llm
        self.grader_llm = grader_llm
        self.answer_generator_llm = answer_generator_llm
        self.question_generators_llm = question_generators_llm
        self.hallucination_llm = hallucination_llm
        self.answer_review_llm = answer_review_llm  
        self.cypher_translator_llm = cypher_translator_llm

    def router_agent(self):
        """Create a router agent for question routing."""
        return Agent(
            role='Router',
            goal='Select a route based on the user question to a vectorstore or web search',
            backstory=dedent("""\
                You are an expert at routing a user question to a vectorstore, web search, or cypher db.
                Use the vectorstore search for questions related to product description, summary, reviews, etc.
                Use the cypher db search for questions related to product inventory, prices, discounts, lead time, shipping, etc.
                Do not be stringent with keywords for these topics; otherwise, use web-search.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.router_llm
        )

    def grader_agent(self):
        """Create a grader agent for assessing document relevance."""
        return Agent(
            role='Retriever_Grader',
            goal='Filter out erroneous retrievals',
            backstory=dedent("""\
                You are a grader assessing the relevance of a retrieved document to a user question.
                If the document contains keywords related to the user question, grade it as relevant.
                Ensure that the answer is relevant to the question without being overly stringent.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.grader_llm
        )

    def answer_generator(self):
        """Create an answer generator agent."""
        return Agent(
            role='Q/A generator',
            goal='Generate the answer based on the question and relevant documents',
            backstory=dedent("""\
                You are an AI language model QA answer generator assistant. Your task is to generate a concise answer based on the given user question and relevant retrieved documents.
                Try to understand the question and generate an accurate answer based on the documents. If unsure, say 'I don’t know.' Use three sentences maximum and keep it concise.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.answer_generator_llm
        )

    def question_generators(self):
        """Create a question generator agent."""
        return Agent(
            role='Question transformer and generator',
            goal='Generate five different questions from the original question',
            backstory=dedent("""\
                You are an AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector database.
                By generating multiple perspectives on the question, you help overcome limitations of distance-based similarity searches. Create rephrased questions that explore different perspectives.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.question_generators_llm
        )

    def hallucination_grader(self):
        """Create a hallucination grader agent."""
        return Agent(
            role='Hallucination Grader',
            goal='Filter out hallucination',
            backstory=dedent("""\
                You are a hallucination grader assessing whether an answer is grounded in or supported by a set of facts.
                Review the response provided to ensure it aligns with the question asked.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.hallucination_llm
        )

    def answer_review_agent(self):
        """Create an answer review agent."""
        return Agent(
            role='Answer Review Specialist',
            goal='Review and refine the response for the user question',
            backstory=dedent("""\
                You are an AI assistant with deep knowledge of user question and answer support. Review the response meticulously to ensure it makes sense for the question asked.
                If the answer already covers everything, return the generation. If the answer is not good, rewrite it concisely. If the response is irrelevant, say 'Sorry, contact customer support.'
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.answer_review_llm
        )

    def cypher_translator(self):
        """Create a Cypher translator agent."""
        return Agent(
            role='Text to Cypher Translator',
            goal='Convert the text questions to cypher database queries',
            backstory=dedent("""\
                You are an expert at converting user questions into cypher database queries. Understand the question and convert it into cypher queries.
            """),
            verbose=True,
            allow_delegation=False,
            llm=self.cypher_translator_llm
        )
