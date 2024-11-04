from crewai import Task
from textwrap import dedent
from .agents import RAG_AGENTS

class RAG_TASKS():
    def __init__(self):
        
        self.router_agent = RAG_AGENTS.router_agent()
        self.grader_agent = RAG_AGENTS.grader_agent()
        self.answer_generator = RAG_AGENTS.answer_generator()
        self.question_generators = RAG_AGENTS.question_generators()
        self.hallucination_grader = RAG_AGENTS.hallucination_grader()
        self.answer_review_agent = RAG_AGENTS.answer_review_agent()
        self.cypher_translator = RAG_AGENTS.cypher_translator()

    def router_task(self):
        return Task(
            description=dedent(f"""
                Analyse the keywords in the question {{question}}.
                Based on the keywords, decide whether it is eligible for a vectorstore search, a web search, or a cypher db search.
                Return 'vectorstore' if eligible for vectorstore search, 'cypher db' for cypher search, and 'websearch' for web search.
                Do not provide any preamble or explanation. Use memory to learn as you go.
            """),
            expected_output=dedent("""
                Provide one choice: 'websearch', 'vectorstore', or 'cypher db' based on the question.
                Do not provide any preamble or explanation.
            """),
            agent=self.router_agent
        )

    def grader_task(self, retriever_task):
        return Task(
            description=dedent(f"""
                Based on the {{documents}} retrieved from the cypher_retriever or vectorstore_retriever for the question {{question}}, evaluate whether the {{document}} is relevant.
            """),
            expected_output=dedent("""
                Binary score 'yes' or 'no' to indicate relevance. 
                Answer 'yes' if the response from the 'retriever_task' aligns with the question asked. 
                Answer 'no' if it does not align. Do not provide preamble or explanations except for 'yes' or 'no'.
            """),
            agent=self.grader_agent
        )

    def answer_generator_task(self):
        return Task(
            description=dedent(f"Generate an answer based on the user question {{question}} using relevant {{documents}}."),
            expected_output="Return a precise and concise answer to the question.",
            agent=self.answer_generator
        )

    def question_generators_task(self):
        return Task(
            description=dedent(f"""
                Generate five different questions based on the user question {{question}}.
                Focus on semantic meaning or intentions and explore different contexts. Include both subjective and objective questions.
            """),
            expected_output="Return rephrased versions of the question.",
            agent=self.question_generators
        )

    def hallucination_grader_task(self):
        return Task(
            description=dedent(f"""
                Evaluate the {{generate}} response from generators for the question {{question}} and retrieved {{documents}} to check if the answer is supported by facts.
            """),
            expected_output=dedent("""
                Binary score 'yes' or 'no' to indicate if the answer aligns with the question. 
                Answer 'yes' if the answer is useful and factual, 'no' if it is not. 
                Do not provide any preamble or explanations except for 'yes' or 'no'.
            """),
            agent=self.hallucination_grader
        )


    def answer_review_task(self):
        return Task(
            description=dedent(f"""
                Review the {{generation}} response drafted for the user {{question}}. 
                Ensure it is comprehensive, accurate, and meets high-quality standards. 
                Check that all parts of the question are addressed with a friendly tone and well-supported references.
            """),
            expected_output=dedent("""
                Return a final, detailed, and informative response ready for the user. 
                Ensure the response is friendly, professional, and maintains the company's tone.
            """),
            agent=self.answer_review_agent
        )

    def cypher_translator_task(self):
        return Task(
            description=dedent(f"""
                Convert the user question {{question}} to cypher database queries. 
                If unfamiliar with acronyms or terms, do not attempt to rephrase.
            """),
            expected_output=dedent("""
                Return a cypher query for database execution. 
                Respond 'Sorry! unable to find a valid response' if not applicable.
            """),
            agent=self.cypher_translator
        )
