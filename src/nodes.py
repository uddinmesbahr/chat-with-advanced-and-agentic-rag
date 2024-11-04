import os
import time
from crew.agents import RAG_AGENTS
from crew.tasks import Tasks
from crewai import Crew
from .retriever import vectorstore_retrieve, cypher_retriever, web_search_tool



class Nodes:
    def __init__(self):
        # Initialize crews
        self.router_crew = Crew(
            agents=[RAG_AGENTS.router_agent],
            tasks=[Tasks.router_task],
            memory=True,
            verbose=True,
        )

        self.grader_crew = Crew(
            agents=[RAG_AGENTS.grader_agent],
            tasks=[Tasks.grader_task],
            verbose=True,
        )

        self.answer_generator_crew = Crew(
            agents=[RAG_AGENTS.answer_generator],
            tasks=[Tasks.answer_generator_task],
            memory=True,
            verbose=True,
        )

        self.question_generators_crew = Crew(
            agents=[RAG_AGENTS.question_generators],
            tasks=[Tasks.question_generators_task],
            verbose=True,
        )

        self.hallucination_crew = Crew(
            agents=[RAG_AGENTS.hallucination_grader],
            tasks=[Tasks.hallucination_grader_task],
            verbose=True,
        )

        self.answer_review_crew = Crew(
            agents=[RAG_AGENTS.answer_review_agent],
            tasks=[Tasks.answer_review_task],
            verbose=True,
        )

        self.cypher_translator_crew = Crew(
            agents=[RAG_AGENTS.cypher_translator],
            tasks=[Tasks.cypher_translator_task],
            verbose=True,
        )

    def router(self, state):
        """Initiates the router agent to generate a routing decision based on the current state."""
        print("---CALL ROUTER AGENT---")
        question = state["question"]
        response = self.router_crew.kickoff(inputs={"question": question})
        return {"question": question, "response": response}

    def route_decision(self, state):
        """Routes the question based on the agent's decision."""
        print("---ROUTE QUESTION DECISION---")
        decision = state["response"]

        if decision == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        elif decision == "vectorstore":
            print("---ROUTE QUESTION TO VECTORSTORE---")
            return "vectorstore"
        elif decision == "cypher db":
            print("---ROUTE QUESTION TO CYPHER DB---")
            return "cypher db"
        else:
            print("---UNKNOWN ROUTE DECISION---")
            return None

    def vectorstore_retrieve(self, state):
        """Retrieves relevant documents for the question from the vectorstore."""
        print("---RETRIEVE DOCUMENTS FROM VECTORSTORE---")
        question = state["question"]

        if isinstance(question, list):
            documents = [self.vectorstore_retriever.get_relevant_documents(q) for q in question]
        else:
            documents = self.vectorstore_retriever.get_relevant_documents(question)

        return {"documents": documents, "question": question}
    
      
    def cypher_translating(self, state):
        """Initiates the router agent to generate a routing decision based on the current state."""
        print("---CALL cypher_translator agent---")
        question = state["question"]
        response = self.cypher_translator_crew.kickoff(inputs={"question": question})
        return {"question": question, "response": response}
    
    def cypher_retriever(self, state):
        """Retrieves relevant documents for the question from the vectorstore."""
        print("---RETRIEVE DOCUMENTS FROM VECTORSTORE---")
        question = state["question"]
        cypher=state["cypher"]

        if isinstance(question, list):
            documents = [self.cypher_retriever(q) for q in cypher]
        else:
            documents = self.cypher_retriever(cypher)

        return {"documents": documents, "question": question}
    
    def web_search(state):
        """
        Web search based on the re-phrased question.

         Args:
            state (dict): The current graph state

         Returns:
             state (dict): Updates documents key with appended web results
         """

        print("---WEB SEARCH---")
        question = state["question"]

        # Web search
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": web_results, "question": question}

    def retrieve_grader(self, state):
        """Checks if the retrieved documents are relevant to the question."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]
        filtered_docs = []

        for doc in documents:
            score = self.grader_crew.kickoff(inputs={"documents": documents, "question": question})
            grade = score["score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    def decide_to_generate(self, state):
        """Decides whether to generate an answer or create multiple new questions."""
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT, TRANSFORM QUERY---")
            return "multiple_question_generators"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    def generate(self, state):
        """Generates an answer based on the question and relevant documents."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]
        generation = self.answer_generator_crew.kickoff(inputs={"documents": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    def multiple_question_generators(self, state):
        """Generates multiple questions for improved document retrieval."""
        print("---GENERATE MULTIPLE QUESTIONS---")
        question = state["question"]
        response = self.question_generators_crew.kickoff(inputs={"question": question})
        return {"questions": response, "original_question": question}

    def reciprocal_rank_fusion(self, state):
        """Performs reciprocal rank fusion on retrieved documents for reranking."""
        print("---FUSION---")
        fusion_documents = state["documents"]
        original_question = state["original_question"]
        documents = []
        fused_scores = {}

        for docs in fusion_documents:
            for rank, doc in enumerate(docs, start=1):
                if doc.page_content not in fused_scores:
                    fused_scores[doc.page_content] = 0
                    documents.append(doc)
                fused_scores[doc.page_content] += 1 / (rank + k)

        reranked_results = {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:3]}
        print("\nTop 3 search scores ========================================")
        for i, (doc, score) in enumerate(reranked_results.items(), start=1):
            print(f"\nDocument {i}: {doc} - Score: {score}")
        print("\n===========================================================\n")

        filtered_documents = [doc for doc in documents if doc.page_content in reranked_results]
        return {"documents": filtered_documents, "question": original_question}

    def hallucination_grader(self, state):
        """Checks for hallucination in the generated response."""
        print("---CHECK HALLUCINATION---")
        question = state["question"]
        generation = state["generation"]
        response = self.hallucination_crew.kickoff(inputs={"question": question, "generation": generation})
        return {"question": question, "multiple_questions": response}

    def decide_after_hallucination_grader(self, state):
        """Decides the next step after checking for hallucination."""
        print("---ASSESS HALLUCINATION GRADER---")
        question = state["question"]
        generation = state["generation"]
        score = self.hallucination_crew.invoke(inputs={"documents": state["documents"], "question": question, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED---")
            return "generate"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED, RETRY---")
            return "Sorry, I am not able to answer this question. Please contact customer service."

    def final_grader(self, state):
        """Evaluates the final generated response to ensure it is grounded and accurate."""
        print("---CHECK FINAL GENERATION---")
        question = state["question"]
        generation = state["generation"]
        score = self.hallucination_crew.invoke(inputs={"documents": state["documents"], "question": question, "generation": generation})
        grade = score["score"]

        if grade == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
            return self.support_answer_quality_assurance.invoke(inputs={"question": question, "generation": generation})
        else:
            print("---DECISION: NOT GROUNDED, RETRY---")
            return "Sorry, I am not able to answer this question. Please contact customer service."

