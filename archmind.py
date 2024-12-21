from typing import List, Tuple, Any, Dict, Optional 
from pydantic import BaseModel, Field

from pyvis.network import Network
from networkx.drawing.nx_pydot import graphviz_layout
import gravis as gv
import networkx as nx
from networkx import DiGraph

import nest_asyncio
import json
from scipy.spatial.distance import cosine
from networkx.readwrite import json_graph

from stores import DesignStore, KnowledgeStore

from llama_index.core import PromptTemplate, QueryBundle, get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever
from llama_index.agent.lats.types import Candidates, SearchNode, Evaluation
from llama_index.core.utils import print_text
from llama_index.core.tools import BaseTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.agent.lats import LATSAgentWorker
from llama_index.core.agent import AgentRunner
from llama_index.core import PromptTemplate
from llama_index.core.agent import AgentChatResponse, Task
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.schema import TextNode

class ContextRetriever(BaseRetriever):

    additional_info = None
    additional_retriever = None

    def __init__(self, context: TextNode, retriever=None) -> None:
        self.context = context
        self.additional_retriever = retriever
        super().__init__()
    
    def set_additional_query(self, query) -> None:
        self.additional_info = query
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve the same nodes irrespective of the given query."""
        my_nodes = [NodeWithScore(node=self.context, score=1.0)]
        if  self.additional_retriever is not None:
            # TODO: Should I also retrieve chunks related to the system context + decisions?
            # my_nodes.extend(self.additional_retriever.retrieve(self.context.text))
            # print("Additional info:", self.additional_info)
            if self.additional_info is not None:
                my_nodes.extend(self.additional_retriever.retrieve(self.additional_info))
            else:
                my_nodes.extend(self.additional_retriever.retrieve(query_bundle.query_str))
        
        return my_nodes

class PatternAssessment(BaseModel):
    decision_name: str = Field(description="name of the decision")
    description: str = Field(description="description of the decision")
    appropriateness: Optional[str] = Field(description="assessment of the decision as appropriate or not for the requirement plus the reasons for this assessment")
    clarifying_questions: Optional[List[str]] = Field(description="a list of additional questions that expert architects might ask about the decision")
    assumptions_and_constraints: Optional[List[str]] = Field(description="a list of assumptions the decision is based on or constraints imposed by the decision")
    qa_consequences: Optional[List[str]] = Field(description="consequences of the decision on system quality attributes")
    risks_and_tradeoffs: Optional[List[str]] = Field(description="a list of potential risks and tradeoffs implied by the decision")
    followup_decisions: Optional[List[str]] = Field(description="a list of technology and implementation decisions that are necessary to implement the decision")

    def __str__(self):
        str_assessment = "Decision: "+ self.decision_name + " - " + self.description
        str_assessment = str_assessment + "\n" + "* Clarifying questions: " + " ".join(self.clarifying_questions)
        str_assessment = str_assessment + "\n" + "* Assumptions and constraints: " + " ".join(self.assumptions_and_constraints)
        str_assessment = str_assessment + "\n" + "* QA consequences: " + " ".join(self.qa_consequences)
        str_assessment = str_assessment + "\n" + "* Risks and tradeoffs: " + " ".join(self.risks_and_tradeoffs)
        str_assessment = str_assessment + "\n" + "* Followup decisions: " + " ".join(self.followup_decisions)
        return str_assessment 
        # f"{self.appropriateness}\n{self.clarifying_questions}\n{self.assumptions_and_constraints}\n{self.qa_consequences}\n{self.risks_and_tradeoffs}\n{self.followup_decisions}"

class PatternCollection(BaseModel):
    patterns: List[str] = Field(description="list of candidate patterns that satisfy a requirement")

class PatternAnalysis(BaseModel):
    pattern_name: str = Field(description="name of the pattern")
    description: str = Field(description="description of the pattern")
    pros: Optional[str] = Field(description="list of advantages of using the pattern")
    cons: Optional[str] = Field(description="list of disadvantages of using the pattern")

    def __str__(self):
        str_pattern = self.pattern_name + " - " + self.description
        str_pattern = str_pattern + "\n" + "* Pros: " + self.pros
        str_pattern = str_pattern + "\n" + "* Cons: " + self.cons
        return str_pattern

class PatternRanking(BaseModel):
    ranking: List[PatternAnalysis] = Field(description="ranking of patterns based on their suitability for a requirement")
    # best_pattern: Optional[str] = Field(description="name of ranked pattern for the requirements")
    # rationale: Optional[str] = Field(description="instantation of the best pattern to satisfy the requirements and rationale for its selection")

class PatternDecision(BaseModel):
    best_pattern: Optional[str] = Field(description="name of best pattern for the requirements")
    rationale: Optional[str] = Field(description="instantation of the best pattern to satisfy the requirements and rationale for its selection")

# class ArchitectureDecisionRecord(BaseModel):
#     title: str = Field(description="short title, representative of the solved problem and found solution by the Architecture Decision Record")
#     context: str = Field(description="context and problem statement for the decision")
#     decision_drivers: str = Field(description="requiremts or other factors that influenced the decision")
#     considered_options: List[str] = Field(description="list of alternative design solutions considered by the experts")
#     decision_outcome: str = Field(description="the best chosen decision from the considered options, including a design justification aligned with the decision drivers that addresses the problem")
#     consequences: List[str] = Field(description="The impact and design considerations regarding the chosen option")
#     pros_cons_options: List[str] = Field(description="Comparison of advantages and disadvantages for each of the considered options")

# class AlternativeSummary(BaseModel):
#     summary: str = Field(description="A concise but descriptive paragraph grounded on the design rationale and reflection for the design solution")
#     key_patterns: List[str] = Field(description="The patterns used by the design solution")

class ArchmindAssistant:

    # TOP_K = 3
    # THRESHOLD = 0.5
    # EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"

    # Return the patterns in JSON format. Only include the names of the patterns in the JSON structure, nothing else.
    PATTERN_SELECTION_PROMPT_TEMPLATE = """
    You are an experienced software architect that assists novice developers to design a software system. 

    The context below describes the main characteristics and operating enviroment for a software system that needs to be developed.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. Remember to focus on the requirements.
    Your main task is to answer with a plain list of up to {n} patterns or architectural tactics that might be applicable for the requirements. 
    Return only the names of the patterns or tactics, nothing else.
    Double-check if the requirements are clear and unambiguous for the provided context. 
    Do not change or alter the original requirements in any way.
    If the requirements do not make sense for the context or are unclear or ambiguous, return an empty list
    and do not try to provide irrelevant patterns or tactics.

    Context information from multiple sources is below.
    ---------------------
    Context: {context_str}

    ---------------------

    Question: What patterns or architectural tactics are applicable for these requirements: ``{query_str}``?

    Answer:"""

    PATTERN_COMPARISON_PROMPT_TEMPLATE = """
    You are an experienced software architect that assists novice developers to design a software system. 

    The context below describes the main characteristics and operating environment for a software system that needs to be developed. 
    Information about a list of candidate patterns or architectural tactics is included.
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. 
    Your main task is to compare and rank the patterns from most to least preferred based on the adequacy of each pattern for the requirements.
    Use only patterns or tactics from the provided list, but you might discard patterns or tactics being unsuitable for any of the requirements.
    For each ranked pattern or tactic, include the following data: 
    - pattern (or tactic) name, 
    - a short description of the pattern (or tactic), 
    - concise pros and cons of using the pattern (or tactic) instantiated on the specific requirements
    If the requirements do not make sense for the context or are unclear or ambiguous do not make any comparison and just return an empty ranking.
    Do not try to compare or return irrelevant patterns (or tactics).

    List of candidate patterns or tactics: {patterns}
    
    Context information from multiple sources is below.
    ---------------------
    Context: {context_str}

    ---------------------

    Question: Can you compare the patterns or architectural tactics above that are applicable for these requirements ``{query_str}``?

    Helpful Answer:"""

    # , which are sorted by their adequacy for the requirements from most preferred to least preferred.
    DECISION_PROMPT_TEMPLATE = """
    You are an experienced software architect that assists novice developers to design a software system. 

    The context below describes the main characteristics and operating environment for a software system that needs to be developed. 
    Information about a list of suitable patterns or architectural tactics is included.
    Each pattern (or tactic) comes with a description of how it can be used to satisfy the requirement, analyzing the pros and cons of using the pattern (or tactic).
    The question at the end is about a list of requirements that this software system must fulfill. 
    The requirements themselves are enclosed by backticks. 
    Your main task is to choose a suitable pattern or tactic (along with its description) from the list in order to satisfy the requirements.
    In addition, briefly explain how the pattern (or tactic) chosen can be instantiated to address the requirements.
    If the requirements do not make sense for the context or are unclear or ambiguous just return 'Requirements or context is unclear'.
    If none of the patterns (or tactics) is relevant to the requirements, just return 'None of the patterns is applicable'.

    List of analyzed patterns: {patterns}
    
    Context information from multiple sources is below.
    ---------------------
    Context: {context_str}

    ---------------------

    Question: Can you select and instantiate a suitable pattern (or architectural tactic) for these requirements ``{query_str}``?

    Helpful Answer:"""

    ASSESSMENT_PROMPT_TEMPLATE = """
    You are an experienced software architect that assists novice developers to design a software system. 

    The context below describes the main characteristics and operating environment
    for a software system that needs to be developed. There is also a list of requirements that this software system must fulfill. 
    For these requirements, a design decision was made that is supposed to address all the requirements. 
    This decision can be expressed as a well-known design pattern, an architectural style, an architectural tactic, or a design principle.
    The question at the end is about this decision, which is enclosed by backticks. 
    Your main task is to help me understand the implications of the decision for the context and requirements provided.
    Please think step by step and assess the following aspects in your answer:
    - whether the design decision is appropriate or not, plus an explanation of your assessment. 
    - a list of clarifying questions that expert architects might ask about the decision in the provided context.
    - assumptions and constraints that the decision is based on.
    - consequences of the decision on quality attributes of the system.
    - potential risks and trade-offs implied by the decision.
    - additional follow-up decisions that are necessary to fully satisfy the requirements.

    Focus on addressing the specific requirements with the decision, rather than on the general use cases for the decision.
    Return the key points of your assessment.
    Carefully check the design decision and do not admit dubious or unknown decisions. 
    If the decision is dubious or unknown just say that and don't perform any assessment.
    If the decision does not make sense for the context or is not appropriate for the requirements, just state what the problem is.

    Requirements to be satisfied by the decision: {requirements}

    Context information from multiple sources is below.
    ---------------------
    Context: {context_str}

    ---------------------

    Question: What is your assessment of the decision ``{query_str}`` for the requirements above?

    Helpful Answer:"""

    ADR_PROMPT_TEMPLATE = """
    You are an experienced software architect that assists novice developers to design a system. 
    
    The context below describes the main characteristics and operating environment for a software system that needs to be developed. 
    As part of the context, there is also a list of target requirements that this software system must fulfill.
    Furthermore, in order to satisfy the requirements, an expert provided a design solution based on patterns and design decisions. 
    This design solution consists of two parts:
    - a design rationale as a list of ordered steps in the form of ReAct thoughts and observations that detail the expert's reasoning to arrive to the design solution.
    - a reflection that assesses how well the design solution satisfies the requirements, including a score from 1 (not-satisfying) to 10 (fully satisfying).

    Your task is to write a summary of {n} lines of text that describes the key aspects of the design solution and its reflection.
    The summary should be grounded on the design rationale and reflection with respect to the target requirements.
    For the design rationale for the design solution, focus mostly on the last two steps of the list that lead to the design solution.
    Consider and include the following additional aspects when creating the summary: 
    - explicitly mention the patterns being used in the design rationale
    - whether the patterns and other decisions are appropriate for the target requirements
    - assumptions and constraints the design solution is based on.
    - consequences of the design solution on quality attributes of the system.
    - potential risks and trade-offs implied by the design solution.
    - additional follow-up decisions that are necessary for the design solution.

    Carefully check the design solution and do not admit dubious or unknown solution or decisions. 
    If any part of the design solution or decisions is dubious or unknown, do not include them in the summary. 
    Return the summary with the contents above and nothing else.
    
    Context information from multiple sources is below.
    ---------------------

    Design rationale provided by the expert: 
    {alternative}

    Reflection and score provided by the expert for assessing the design solution: 
    {reflection}

    ---------------------

    Task: Provide a paragraph capturing the rationale of the solution and its reflection for the requirement ``{query_str}``.

    Helpful Answer:"""

    def __init__(self, sys_store: KnowledgeStore, dk_store: DesignStore, llm, sys_id=None, rag=False) -> None:
        self.llm = llm
        self.sys_store = sys_store
        self.dk_store = dk_store
        self.sys_id = sys_id
        self.rag_mode = rag
        
        self.reset()
    
    def reset(self):
        self.context = None
        self.requirement = None
        self.patterns = []
        # self.decision = None
        # self.assessment = None
        # self.prior_decisions = {} # Dictionary of temporary decisions (not coming from the database)

        self.retriever = None
    
    def rag_enabled(self):
        return self.rag_mode
    
    def configure_retriever(self, collection: str=None, top_k: int=3, threshold: float=0.7) -> None:
        self.retriever= self.dk_store.get_retriever(collection=collection, threshold=threshold, k=top_k)
        self.threshold = threshold

    def fetch_context(self, with_decisions=False) -> str|None:
        context_node = self.sys_store.get_system() if (self.sys_id is not None) else None
        self.context = context_node.text if (context_node is not None) else None
        if not with_decisions:
            return self.context
        else:
            node = self._create_context_node(self.context)
            return node.text
    
    def fetch_requirement(self, query: str) -> str|None:
        requirement_node = self.sys_store.get_requirement(query) if (query is not None) else None
        self.requirement = requirement_node.text if (requirement_node is not None) else query
        if self.requirement != query:
            print(">>> from database:", self.requirement)
        return self.requirement

    # def fetch_decision(self, query: str) -> str|None:
    #     decision_node = self.sys_store.get_decision(query) if (query is not None) else None
    #     self.decision = decision_node.text if (decision_node is not None) else query
    #     return self.decision

    def add_decision(self, id:str, decision: str):
        if id in self.prior_decisions.keys():
            print("Warning: decision "+str(id)+" already exists, it will be overriden")
        self.prior_decisions[id] = decision

    def get_decision(self, id:str):
        if id in self.prior_decisions.keys():
            return self.prior_decisions[id]
        else:
            return None

    def get_decisions(self):
        return [k+": "+d for k,d in self.prior_decisions.items()]

    def remove_decision(self, id:str):
        d = self.prior_decisions.pop(id, None)
        return d

    def clear_decisions(self):
        self.prior_decisions = {}

    def _create_context_node(self, context:str) -> TextNode:
        my_context_text = None
        if self.context is not None:
            my_context_text = self.context
        elif context is not None:
            print("Creating a context node:", context, self.sys_id)
            my_context_text = context
        else:
            print("Creating a empty context node:", "", self.sys_id)
            my_context_text = ""
        
        # TODO: Should I add the decisions here?
        decisions = self.get_decisions()
        if len(decisions) > 0:
            decisions_message = "\n\nDesign decisions made already include: \n* "+"\n* ".join(decisions)
            my_context_text = my_context_text + decisions_message
            # print("Context with decisions:", my_context_text)
        
        my_context_node = TextNode(text=my_context_text, id=self.sys_id)
        return my_context_node

    # 1. Search for candidate patterns that satisfy the given requirement
    def find_patterns(self, requirement:str, context:str=None, n=5, use_database=False, raw=False) -> List[str]|PatternCollection:
        my_context_node = self._create_context_node(context) # self.sys_store.get_system() if (self.sys_id is not None) else None
        my_requirements = str([requirement])  # str([self.fetch_requirement(requirement)])

        # If use_database=True, then get the patterns directly from the database (without calling the LLM)
        # Applicable only when the requirement ID is known and exists in the database
        if use_database:
            retrieved_patterns = self.sys_store.get_patterns_for_requirement(requirement)
            retrieved_patterns = [p.text for p in retrieved_patterns]
            print("Patterns retrieved from database:", retrieved_patterns)
            if len(retrieved_patterns) > 0:
                return retrieved_patterns

        custom_retriever = ContextRetriever(my_context_node, self.retriever)
        # custom_retriever.set_additional_query(query=my_requirements)

        my_template = PromptTemplate(ArchmindAssistant.PATTERN_SELECTION_PROMPT_TEMPLATE)
        # prompt = my_template.format(context=my_context, n=n)
        # synth = get_response_synthesizer(text_qa_template=sys_template)
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=self.threshold)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, # node_postprocessors=[postprocessor], 
            text_qa_template=my_template.partial_format(n=n), 
            llm=self.llm, # response_synthesizer=synth, 
            output_cls=PatternCollection, verbose=True
        )

        query_text = my_requirements + " "+ " \n".join(self.get_decisions())
        # print(query_text)
        nodes = query_engine.retrieve(query_text)
        if (len(nodes) <= 1) and self.rag_mode:
            print(len(nodes), "chunks retrieved (rag). No patterns found, as there's no grounding in knowledge database!")
            return [] # No ranking generated (no grounding in database)
        
        rag_message = "(rag)" if self.rag_mode else "(zero-shot)"
        print(len(nodes), "chunks retrieved", rag_message)
        response = query_engine.query(my_requirements)
        # print(response)
        if not isinstance(response.response, PatternCollection):
            print(response.response)
            return []
        if raw:
            return response.response
        else:
            return response.response.patterns

    # 2. Compare and rank a list of patterns for a given requirement
    def compare_patterns(self, requirement: str, patterns: List[str], context:str=None, raw=False) -> List[str]|PatternRanking: 
        my_context_node = self._create_context_node(context) # self.sys_store.get_system() if (self.sys_id is not None) else None
        my_requirements = str([requirement])  # str([self.fetch_requirement(requirement)])
        my_patterns = str(patterns)

        custom_retriever = ContextRetriever(my_context_node, self.retriever)
        custom_retriever.set_additional_query(query=my_patterns)

        my_template = PromptTemplate(ArchmindAssistant.PATTERN_COMPARISON_PROMPT_TEMPLATE)
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=self.threshold)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, # node_postprocessors=[postprocessor], 
            text_qa_template=my_template.partial_format(patterns=my_patterns), 
            llm=self.llm, 
            output_cls=PatternRanking, verbose=True
        )
        
        query_text = my_requirements + " "+ " \n".join(self.get_decisions())
        # print(query_text)
        nodes = query_engine.retrieve(query_text)
        if (len(nodes) <= 1) and self.rag_mode:
            print(len(nodes), "chunks retrieved (rag). No pattern comparison, as there's no grounding in knowledge database!")
            return [] # No ranking generated (no grounding in database)
        
        rag_message = "(rag)" if self.rag_mode else "(zero-shot)"
        print(len(nodes), "chunks retrieved", rag_message)
        response = query_engine.query(my_requirements)
        if raw:
            return response.response
        else:
            ranking_results = [str(p) for p in response.response.ranking]
            return ranking_results #(ranking_results, response.response.rationale)
    
    # 3. Select a pattern and instantiate it for a given requirement (short form of decision)
    def make_decision(self, requirement: str, decisions: List[str], context:str=None, raw=False) -> str|PatternDecision:
        my_context_node = self._create_context_node(context) # self.sys_store.get_system() if (self.sys_id is not None) else None
        my_requirements = str([requirement])  # str([self.fetch_requirement(requirement)])
        my_decisions = str(decisions)

        custom_retriever = ContextRetriever(my_context_node, self.retriever)
        custom_retriever.set_additional_query(query=my_decisions)

        my_template = PromptTemplate(ArchmindAssistant.DECISION_PROMPT_TEMPLATE)
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=self.threshold)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, # node_postprocessors=[postprocessor], 
            text_qa_template=my_template.partial_format(patterns=my_decisions), 
            llm=self.llm, 
            output_cls=PatternDecision, verbose=True
        )

        query_text = my_requirements + " "+ " \n".join(self.get_decisions())
        # print(query_text)
        nodes = query_engine.retrieve(query_text)
        if (len(nodes) <= 1) and self.rag_mode:
            print(len(nodes), "chunks retrieved (rag). No decision made, as there's no grounding in knowledge database!")
            return None # No decision made (no grounding in database)
       
        rag_message = "(rag)" if self.rag_mode else "(zero-shot)"
        print(len(nodes), "chunks retrieved", rag_message)
        # print(len(nodes), "chunks retrieved (zero-shot anyway)")
        response = query_engine.query(my_requirements)
        if raw:
            return response.response
        else:
            # print(response.response.best_pattern, response.response.rationale)
            return response.response.best_pattern + " - " + response.response.rationale

    # 4. Analyze in detail a decision (instantiated pattern) for a given requirement
    def assess_decision(self, requirement: str, decision: str, context:str=None, raw=False) -> str|PatternAssessment:        
        my_context_node = self._create_context_node(context) # self.sys_store.get_system() if (self.sys_id is not None) else None
        my_requirements = str([requirement]) #str([self.fetch_requirement(requirement)])
        my_decision = decision

        custom_retriever = ContextRetriever(my_context_node, self.retriever)
        # custom_retriever.set_additional_query(query=my_decision)

        my_template = PromptTemplate(ArchmindAssistant.ASSESSMENT_PROMPT_TEMPLATE)
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=self.threshold)
        query_engine = RetrieverQueryEngine.from_args(
            retriever=custom_retriever, # node_postprocessors=[postprocessor], 
            text_qa_template=my_template.partial_format(requirements=my_requirements), 
            llm=self.llm, 
            output_cls=PatternAssessment, verbose=True
        )

        query_text = my_decision + " "+ " \n".join(self.get_decisions())
        # print(query_text)
        nodes = query_engine.retrieve(query_text)
        if (len(nodes) <= 1) and self.rag_mode:
            print(len(nodes), "chunks retrieved (rag). No decision assessment, as there's no grounding in knowledge database!")
            return None # No decision assessed (no grounding in database)
        
        rag_message = "(rag)" if self.rag_mode else "(zero-shot)"
        print(len(nodes), "chunks retrieved", rag_message)
        response = query_engine.query(my_decision)
        if raw:
            return response.response
        else:
            # Convert response to string
            return str(response.response)
    
    # Summarize the main characteristics of a given solution
    def get_solution_summary(self, requirement: str, trajectory: str, reflection: str, n:int=10, context:str=None, raw=False) -> str:
        # my_context_node = self._create_context_node(context) # self.sys_store.get_system() if (self.sys_id is not None) else None
        my_requirements = str([requirement]) #str([self.fetch_requirement(requirement)])

        chat_engine = SimpleChatEngine.from_defaults(llm=self.llm)
        my_template = PromptTemplate(ArchmindAssistant.ADR_PROMPT_TEMPLATE)
        t = my_template.format(query_str=my_requirements, n=n, alternative=trajectory, reflection=reflection)
        # print(t)
        response = chat_engine.chat(t)
        #print(response)
        return response.response
    
    # def get_summary(self, requirement: str, n:int, trajectory: str, reflection: str, context:str=None, raw=False) -> str:
    #     my_context_node = self.sys_store.get_system() if (self.sys_id is not None) else None
    #     my_requirements = str([requirement]) #str([self.fetch_requirement(requirement)])

    #     custom_retriever = ContextRetriever(my_context_node)
    #     my_template = PromptTemplate(ArchmindAssistant.ADR_PROMPT_TEMPLATE)
    #     t = my_template.partial_format(n=n, alternative=trajectory, reflection=reflection)

    #     # TODO: Cambiar a un completion comun de un chat
    #     query_engine = RetrieverQueryEngine.from_args(
    #         retriever=custom_retriever,
    #         text_qa = t,
    #         llm=self.llm,
    #         output_cls=AlternativeSummary, verbose=True
    #     )
    #     print(my_template.format(n=n, alternative=trajectory, reflection=reflection))
        
    #     nodes = query_engine.retrieve(my_requirements)
    #     print(len(nodes), "chunks retrieved")
    #     response = query_engine.query(my_requirements)
        
    #     if raw:
    #         return response.response
    #     else:
    #         return response.response


class LATSDesignAgentWorker(LATSAgentWorker):

    # Overrides method in super class
    async def _get_next_candidates(self, cur_node: SearchNode, input: str) -> List[str]:
        """Get next candidates."""
        # get candidates
        history_str = "\n".join([s.get_content() for s in cur_node.current_reasoning])

        candidates = await self.llm.astructured_predict(
            Candidates,
            prompt=self.candiate_expansion_prompt,
            query=input,
            conversation_history=history_str,
            num_candidates=self.num_expansions,
        )
        candidate_strs = candidates.candidates[: self.num_expansions]
        if self.verbose:
            print_text(f"> Got {len(candidate_strs)} candidates: {candidate_strs}\n", color="yellow")
        
        # TODO: Check the intent of the thought candidates? (e.g., using a semantic router)
        # print("HERE")
        # ensure we have the right number of candidates
        # if len(candidate_strs) < self.num_expansions:
        #     return (candidate_strs * self.num_expansions)[: self.num_expansions]
        # else:
        #     return candidate_strs[: self.num_expansions]
        return candidate_strs
    

class ArchmindDesignAgent:

    EXPAND_CANDIDATES_PROMPT = """
    You are an experienced software architect that assists novice developers to design a system. 
    Focus on the latest thought, action, and observation of the conversation trajectory.
    Your task is to generate candidates for the next reasoning step.

    # Instructions for generating candidates:
    -----------------------------------------
    * Given a query and a conversation trajectory, generate by default only one candidate thought for the next reasoning step.
    * A candidate can refer to any of the following goals:
    - find a list of design patterns, architectural tactics or follow-up design decisions that could satisfy a given requirement
    - compare a list of one or more concrete design patterns, architectural tactics or follow-up design decisions, which must have been identified in a previous observation.
    - make a design decision that applies and instantiates a concrete design pattern or architectural tactic, which must have been analyzed in a previous observation. 
    - perform an assessment of a design decision that was made in a previous reasoning step (e.g., an existing observation) and was based on a concrete design pattern.
    * As an exception, only if the conversational history includes outputs or observations from the actions 'find_patterns' or 'compare_patterns', 
    you can pick up to {num_candidates} patterns, styles, tactics or design decisions explicitly mentioned in the history \
    and generate one candidate per pattern or decision for the next reasoning step.
    * Do not generate candidates about assessment of decisions or patterns if they are not mentioned in the conversation history.
    * The candidate thoughts being generated must be always different; avoiding generating different candidates that share the same semantic meaning.
    * Keep the abstraction level of the design decisions at a high-level or middle-level of architecture design, avoiding low-level details (e.g., code snippets or object-oriented classes).

    Do not generate additional thoughts or actions.

    Query: {query}
    Conversation History:
    {conversation_history}
    """

    REFLECTION_PROMPT = """
    You are an experienced software architect that assists novice developers to design a system. 

    Given a query and a ReAct conversation trajectory, your main task is to evaluate 3 criteria regarding whether the conversation answers the user question:
    - **correctness**: Whether the thoughts and actions so far are correctly addressing the question, even if the answer is not found yet. Rate this criterion from 1-10, where 10 is fully correct.
    - **completeness**: Whether the answer is found yet. Rate this criterion as 1 if the answer was found and 0 otherwise.
    - **quality of the design decisions**: Whether the design decisions (e.g., patterns) made satisfy all the requirements or additional decisions are necessary. Rate this criterion from 1-10, where 10 means that all necessary decisions are present and all the requirements are met.
    Provide your reasoning and analysis in detail, along with a global score in the range of 1 (worst) to 10 (best), based on the scores for the individual criteria above.
    In any case in which a design decision or pattern or tactic has been applied, having a detailed assessment of it is important to achieve a good score.
    Focus on the latest thought, action, and observation from the trajectory.
    Incomplete trajectories can be correct if the thoughts and actions so far are correct, even if the answer is not found yet.
    Do not generate additional thoughts or actions.

    Query: {query}
    Conversation History:
    {conversation_history}
    """

    SYSTEM_PROMPT = """
    Your are an assistant that helps software architects to design software systems by providing them 
    with the necessary knowledge to arrive to a design solution. When you are provided with a requirement, 
    you can access different tools for selecting candidate patterns, architectural styles or tactics for satisfying the requirement, 
    comparing those patterns in order to make one or more decisions, and assessing each decision in detail.
    Always respond in English and elaborate a comprehensive final answer about the solution.
    """

    QUESTION_PROMPT = """\
    Explore design alternatives and make one or more final, assessed decisions in order to obtain \
    a design solution that fully satisfies the requirement: ``{requirement}``. \
    Once a decision is made with the corresponding tool, the decisions must be assessed with the appropriate tool. \
    Address the requirements using a unique trajectory of thoughts, actions, and observations. This is trajectory #{trajectory}. \
    Do not mention the reflections nor the scores about trajectories in the final answer.
    There can be previous design decisions or solutions that can satisfy the requirement to some degree, \
    which MUST be taken into account during the reasoning process to avoid exploring similar solutions.
    {existing_decisions}

    """

    HISTORY_PROMPT = """\
    ALWAYS try decisions (e.g., patterns, tactics, styles) that are DIFFERENT from the solutions from previous trajectories. \
    Each trajectory includes a score regarding the solution adequacy for the requirements along with a reflection.
    {history}
    """

    def __init__(self, assistant: ArchmindAssistant, llm=None) -> None:
        self.assistant = assistant
        self.agent = None
        self.llm = llm
    
    def find_patterns(self, requirement: str) -> str:
        """
        Useful for finding candidate patterns applicable to the given requirement.

        Args:
            requirement (str): The requirement for which patterns need to be selected.

        Returns:
            str: A list of candidate patterns that can satisfy the given requirement.
        """

        patterns = self.assistant.find_patterns(requirement=requirement)
        if len(patterns) > 0:
            return "Candidate patterns for requirement '"+requirement+"': ["+", ".join(patterns)+"]"
        else:
            return "No candidate patterns found for requirement '"+requirement+"'"

    def compare_patterns(self, requirement: str, candidate_patterns: List[str]) -> str:
        """
        Useful for analizing, comparing and shortlisting patterns for a given requirement.

        Args:
            requirement (str): The requirement for which patterns need to be compared.
            patterns (List[str]): A list of candidate patterns to be analyzed and compared.

        Returns:
            str: A list of analyzed patterns (from most to least preferred) for the given requirement.
        """

        pattern_ranking = self.assistant.compare_patterns(requirement=requirement, patterns=candidate_patterns)
        if len(pattern_ranking) == 0:
            return "No valid patterns could be compared for requirement '"+requirement+"'"
        else:
            return "Suitable patterns for requirement '"+requirement+"': [\n"+", \n".join(pattern_ranking)+"\n]"

    def make_decision(self, requirement: str, candidate_patterns: List[str]) -> str:
        """
        Useful for selecting an initial design decision suitable for a given requirement, once a list of analyzed patterns is available.
        The decision can also refer to a follow-up decision resulting from a pattern previously applied for the same requirement.

        Args:
            requirement (str): The requirement for which a decision needs to be made.
            candidate_patterns (List[str]): A list of candidate patterns or decisions (from most to least preferred) to be compared.

        Returns:
            str: The design decision and its rationale regarding the given requirement. Once made, this decision must be assessed in detail.
        """

        decision = self.assistant.make_decision(requirement=requirement, decisions=candidate_patterns)
        # print(decision)
        if decision is not None:
            return "Initial design decision for requirement '"+requirement+"': "+ decision
        else:
            patterns_list = "["+", ".join(candidate_patterns)+"]"
            return "No valid design decision could be made for requirement '"+requirement+"'. However, you might still try to assess any of "+patterns_list+" as a general pattern for the problem."

    def assess_decision(self, requirement: str, decision: str) -> str:
        """
        Useful for assessing an initial design decision for a given requirement and then generating a detailed assessment.

        Args:
            requirement (str): The requirement for which the design decision was made.
            decision (str): The initial design decision made for the requirement.

        Returns:
            str: A detailed assessment of the design decision, which might trigger further design work or actions.
        """

        assessment = self.assistant.assess_decision(requirement=requirement, decision=decision)
        if assessment is not None:
            return "Assessment for design decision '"+decision+"' for requirement '"+requirement+"': "+ assessment
        else:
            return "No assessment was possible for design decision '"+decision+"' for requirement '"+requirement+"'"

    def create_agent(self, mode:str = 'react', **args) -> ReActAgent|AgentRunner:
        """Create an agent for the given mode."""
        
        find_patterns_tool = FunctionTool.from_defaults(fn=self.find_patterns)
        compare_patterns_tool = FunctionTool.from_defaults(fn=self.compare_patterns)
        make_decision_tool = FunctionTool.from_defaults(fn=self.make_decision)
        assess_decision_tool = FunctionTool.from_defaults(fn=self.assess_decision)
        
        tools = [find_patterns_tool, compare_patterns_tool, make_decision_tool, assess_decision_tool]

        if mode == 'react':
            self.agent = ReActAgent.from_tools(tools, llm=self.llm, # max_iterations=5,
                                                context=ArchmindDesignAgent.SYSTEM_PROMPT, **args)
            return self.agent
        if mode == 'lats':
            nest_asyncio.apply()
            expand_candidates_prompt = PromptTemplate(ArchmindDesignAgent.EXPAND_CANDIDATES_PROMPT)
            reflection_prompt = PromptTemplate(ArchmindDesignAgent.REFLECTION_PROMPT)
            agent_worker = LATSDesignAgentWorker.from_tools( #LATSAgentWorker.from_tools(
                tools=tools,
                llm=self.llm,
                # num_expansions=3,
                # max_rollouts=5,  # using -1 for unlimited rollouts
                # verbose=True,
                candiate_expansion_prompt=expand_candidates_prompt,
                reflection_prompt=reflection_prompt,
                context=ArchmindDesignAgent.SYSTEM_PROMPT, **args
            )
            self.agent = AgentRunner(agent_worker)
            return self.agent
        
        print("Warning: Mode not recognized")
        self.agent = None
        return self.agent

    def reset_agent(self):
        self.agent.reset()

    def reflect_on_task(self, task: Task) -> Evaluation:
        all_reasoning = task.extra_state['current_reasoning']
        history_str = "\n".join([s.get_content() for s in all_reasoning])
        reflection_prompt = PromptTemplate(ArchmindDesignAgent.REFLECTION_PROMPT)
        evaluation = self.llm.structured_predict(Evaluation, prompt=reflection_prompt, query=input,conversation_history=history_str)

        return evaluation
    
    @staticmethod
    def filter_redundant_alternatives(alternatives: List[str], similarity_threshold=0.8) -> List[str]:
        nodes = [TextNode(id_=str(a), text=text) for a,text in enumerate(alternatives)]
        index = VectorStoreIndex(nodes)
        # index.vector_store.data.embedding_dict.keys()
        unique_texts = []
        seen_docs = []
        for doc in nodes:
            is_similar = False
            doc_vector = index.vector_store.data.embedding_dict[doc.id_]
            for seen_doc in seen_docs:
                another_vector = index.vector_store.data.embedding_dict[seen_doc.id_]
                threshold = 1-cosine(doc_vector, another_vector)
                # print(threshold, doc)
                # print("---", seen_doc)
                # print("=="*10)
                if threshold > similarity_threshold:
                    is_similar = True
                    break
            if not is_similar:
                unique_texts.append(doc.text)
                seen_docs.append(doc)

        # Output the unique texts
        return unique_texts
    
    def get_alternatives(self, requirement, graph=None, n=10, filter_redundancy=True) -> List[str]:
        if graph is None:
            print("Getting graph from agent!")
            graph = self.save_agent_state(as_graph=True, alternatives_only=True)
        if graph is not None:
            print("Summarizing (and sorting)", graph.number_of_nodes(), "alternatives...")
            trajectories = [str(data['current_reasoning']) for n,data in graph.nodes(data=True)]
            reflections = [str(data['evaluation']) for n,data in graph.nodes(data=True)]
            scores = [data['evaluation']['score'] for n,data in graph.nodes(data=True)]
            alternatives = []
            for t,r,score in zip(trajectories, reflections, scores):
                # print("========")
                # alt = self.assistant.get_summary(requirement, n=10, trajectory=t, reflection=r)
                alt = self.assistant.get_solution_summary(requirement, trajectory=t, reflection=r, n=n)
                # print("ALTERNATIVE:", alt)
                alternatives.append((score,alt))
            alternatives.sort(key=lambda x: x[0], reverse=True)
            result = [tuple[1] for tuple in alternatives]
            print("Raw list of alternatives:", len(result))
            if filter_redundancy:
                print("  filtering redundant alternatives...")
                return ArchmindDesignAgent.filter_redundant_alternatives(result)
        return []

    def run_agent(self, requirement: str, n_cycles=1, avoid_redundancy=True) -> str: #AgentChatResponse:
        
        response = None
        evaluation = None
        for i in range(1, n_cycles+1):

            print_text("\nCYCLE "+str(i)+" ==========", color='pink', end='\n')
            
            previous_decisions = self.assistant.get_decisions()
            decisions_message = ""
            if len(previous_decisions) > 0:
                decisions_message = "\nIn this case, the design decisions or solutions applied already include: \n* "+"\n* ".join(previous_decisions)
            question_i = PromptTemplate(ArchmindDesignAgent.QUESTION_PROMPT).format(requirement=requirement,trajectory=i, existing_decisions=decisions_message)
            
            if response is not None:
                history_message = "Trajectory #"+str((i-1))+": "+response.response
                if evaluation is not None:
                    history_message = history_message + "\nReflection for trajectory #"+str((i-1))+": score="+str(evaluation.score)+", "+evaluation.reasoning
                history = PromptTemplate(ArchmindDesignAgent.HISTORY_PROMPT).format(history=history_message)
                question_i = question_i + "\n" + history
            # print(question_i)
            print()
            response = self.agent.chat(question_i)
            print_text("\nRESPONSE "+str(i)+": "+response.response, color='pink', end='\n')

            if isinstance(self.agent, ReActAgent):
                task = self.agent.list_tasks()[-1] # last task
                print_text("\nSelf-reflection on task "+str(task.task_id)+" ...", color='pink', end='\n')
                evaluation = self.reflect_on_task(task) 
                reflection_dict = {}
                reflection_dict['score'] = evaluation.score
                reflection_dict['is_done'] = evaluation.is_done
                reflection_dict['reasoning'] = evaluation.reasoning
                task.extra_state['reflection'] = reflection_dict
            
        print_text("\n====================", color='pink', end='\n')
        
        list_of_alternatives = self.get_alternatives(requirement, filter_redundancy=avoid_redundancy) 
        # Summarization of the alternatives (trajectories + reflections) in the response
        return list_of_alternatives

    @staticmethod
    def _generate_single_node_graph(react_graph, current_reasoning: List, task_id=None, reflection_dict=None):

        root_node_dict = {}
        objs = [json.loads(o.model_dump_json()) for o in current_reasoning]
        root_node_dict['current_reasoning'] = objs #[json.dump(o) for o in objs]
        if task_id is not None:
            root_node_dict['task_id'] = task_id
        if reflection_dict is not None:
            root_node_dict['is_done'] = reflection_dict['is_done']
            root_node_dict['score'] = reflection_dict['score']
            root_node_dict['evaluation'] = reflection_dict
        id_ = id(current_reasoning)
        react_graph.add_node(id_, **root_node_dict)

        return react_graph

    @staticmethod
    def _generate_tree_graph(mcts_graph, root_node: SearchNode, task_id=None):

        root_node_dict = {}
        objs = [json.loads(o.model_dump_json()) for o in root_node.current_reasoning]
        # print(objs)
        root_node_dict['current_reasoning'] = objs #[json.dump(o) for o in objs]
        root_node_dict['evaluation'] = json.loads(root_node.evaluation.model_dump_json())
        root_node_dict['visits'] = root_node.visits
        root_node_dict['answer'] = root_node.answer
        if task_id is not None:
            root_node_dict['task_id'] = task_id
        root_node_dict['score'] = root_node.score
        root_node_dict['is_done'] = root_node.is_done
        id_ = id(root_node)
        mcts_graph.add_node(id_, **root_node_dict)

        nodes_to_visit = [root_node]
        ids = [id_]
        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            parent_id = ids.pop(0)
            # print("Visiting: ", parent_id)
            for child in node.children:
                node_dict = {}
                node_dict['current_reasoning'] = [json.loads(o.model_dump_json()) for o in child.current_reasoning]
                node_dict['evaluation'] = json.loads(child.evaluation.model_dump_json())
                node_dict['visits'] = child.visits
                node_dict['answer'] = child.answer
                node_dict['score'] = child.score
                node_dict['is_done'] = child.is_done
                id_ = id(child)
                mcts_graph.add_node(id_, **node_dict)
                mcts_graph.add_edge(parent_id, id_)
                # print("Adding edge from", parent_id, "to", id_)
                nodes_to_visit.append(child)
                ids.append(id_)

        return mcts_graph
    
    def save_agent_state(self, filename: str=None, indent: int=4, as_graph=False, alternatives_only=False):
        if self.agent is None:
            return None
        
        graph = nx.DiGraph()
        if isinstance(self.agent, ReActAgent):
            graph = nx.DiGraph()
            tasks = self.agent.list_tasks()
            for task in tasks:
                # print(task)
                graph = ArchmindDesignAgent._generate_single_node_graph(graph, 
                            task.extra_state['current_reasoning'], task.task_id, task.extra_state['reflection'])
            print(graph.number_of_nodes(), "nodes in the graph")

        if isinstance(self.agent.agent_worker, LATSDesignAgentWorker):
            # If alternative is enabled, then persist only the leaves of the tree
            graph = nx.DiGraph()
            tasks = self.agent.list_tasks()
            for task in tasks:
                # print(task)
                graph =  ArchmindDesignAgent._generate_tree_graph(graph, task.extra_state['root_node'], task.task_id)
            if alternatives_only:
                nodes_to_remove = [n for n in graph.nodes() if graph.out_degree(n) > 0]
                graph.remove_nodes_from(nodes_to_remove) 
                for n, data in graph.nodes(data=True):
                    data['task_id'] = task.task_id       
            print(graph.number_of_nodes(), "nodes in the graph")
        
        if graph is not None:
            obj = json_graph.node_link_data(graph)
            if filename is not None:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(obj, f, ensure_ascii=False, indent=indent)
            if as_graph:
                return graph
            return json.dumps(obj, indent=indent)
        
        return None


class DesignReasoningVisualizer():

    def __init__(self, graph: DiGraph) -> None:
        self.graph = graph

    @staticmethod
    def read_from_json(filename: str, alternatives_only=False) -> DiGraph:
        with open(filename) as f:
            data = json.load(f)
        graph = json_graph.node_link_graph(data, directed=True)
        if alternatives_only:
            nodes_to_remove = [n for n in graph.nodes() if graph.out_degree(n) > 0]
            graph.remove_nodes_from(nodes_to_remove)  
            print(graph.number_of_nodes(), "nodes in the graph")
        return graph

    @staticmethod
    def _add_graph_layout(graph: DiGraph, invert=True):
        
        copy_graph = nx.DiGraph()
        for m,n in graph.edges():
            copy_graph.add_edge(m,n)

        # pos = nx.drawing.layout.spring_layout(graph, scale=600)
        # pos = nx.drawing.layout.kamada_kawai_layout(graph, scale=600)
        # pos = nx.drawing.layout.spiral_layout(graph, scale=600)
        pos = graphviz_layout(copy_graph, prog="dot")

        # Add coordinates as node annotations that are recognized by gravis
        max_y = 0
        for name, (x, y) in pos.items():
            node = copy_graph.nodes[name]
            graph.nodes[name]['x'] = x
            graph.nodes[name]['y'] = invert*y
            if y > max_y:
                max_y = y
    
        copy_graph.clear()
        # print("Max Y:", max_y)
        return graph, pos
      
    @staticmethod  
    def _get_step_color(steps):
        if 'response' in steps.keys():
            return 'green'
        if 'thought' in steps.keys():
            return 'yellow'
        if 'action' in steps.keys():
            return 'cyan'
        if 'observation' in steps.keys():
            return 'blue'
        return 'gray'

    @staticmethod  
    def _get_step_hover_text(steps):
        if 'response' in steps.keys():
            return steps['response']
        if 'action' in steps.keys():
            return steps['thought']+"\naction "+steps['action']+": "+str(steps['action_input'])
        if 'thought' in steps.keys():
            return steps['thought']
        if 'observation' in steps.keys():
            return steps['observation']
        return ""

    @staticmethod  
    def _get_node_hover_text(data):
        score_message = ""
        if 'score' in data:
            score_message = "score={:.2f}".format(round(data['score'], 2))
            if 'evaluation' in data:
                justification = data['evaluation']['reasoning']
                if len(justification) > 0:
                    score_message = justification+"\n"+score_message
            if 'task_id'in data:
                taskid_message = "task_id: "+str(data['task_id'])
                score_message = taskid_message if score_message is None else taskid_message+"\n"+score_message
        return score_message

    @staticmethod
    def _get_step_label(steps, node_id):
        action_name = 'action'
        if 'response' in steps.keys():
            return 'response_'+node_id
        if 'action' in steps.keys():
            action_name = steps['action']
        if 'thought' in steps.keys():
            return 'thought_'+action_name+'_'+node_id
        if 'observation' in steps.keys():
            return 'observation_'+node_id
        return 'undefined_'+node_id

    @staticmethod  
    def _get_internal_nodes(graph: DiGraph, node, reasoning: List, filter_history=True):
        preds = list(graph.predecessors(node))
        if filter_history:
            parent_node = preds[0] if len(preds) > 0 else None
            n =  graph.nodes[parent_node]['number_steps'] if parent_node is not None else 0
            filtered_reasoning = reasoning[n:len(reasoning)]
            # print("\tremoving:", n, "result:", len(filtered_reasoning))
        else:
            filtered_reasoning = reasoning
        previous_node = node
        for idx, r in enumerate(filtered_reasoning):
            id_ = str(node)+"_"+str(idx)
            id_ = DesignReasoningVisualizer._get_step_label(r, id_)
            node_dict = {}
            node_dict['internal'] = True
            node_dict['step'] = r
            node_dict['color'] = DesignReasoningVisualizer._get_step_color(r)
            node_dict['hover'] = DesignReasoningVisualizer._get_step_hover_text(r)
            node_dict['shape'] = 'circle'
            node_dict['size'] = 15
            node_dict['border_color'] = 'black'
            node_dict['label_size'] = 12
            node_dict['border_size'] = 1
            node_dict['node_hover_tooltip'] = True
            graph.add_node(id_, **node_dict)
            graph.add_edge(previous_node, id_, size=1.5)
            previous_node = id_
        return graph

    @staticmethod
    def _decorate_graph(graph, expand_reasoning=False, filter_history=True):
        copy_graph = graph.copy()
        copy_graph = nx.convert_node_labels_to_integers(copy_graph) #, ordering='increasing degree')

        # Default characteristics for the graph
        nx.set_node_attributes(copy_graph, False, 'internal')
        nx.set_node_attributes(copy_graph, 'gray', 'color')
        nx.set_node_attributes(copy_graph, 'rectangle', 'shape')
        nx.set_node_attributes(copy_graph, 25, 'size')
        nx.set_node_attributes(copy_graph, 'black', 'border_color')
        nx.set_node_attributes(copy_graph, True, 'node_hover_neighborhood')
        nx.set_node_attributes(copy_graph, True, 'node_hover_tooltip')
        nx.set_node_attributes(copy_graph, 12, 'label_size')
        nx.set_node_attributes(copy_graph, 1, 'border_size')
        nx.set_edge_attributes(copy_graph, 3, 'size')

        initial_nodes = list(copy_graph.nodes(data=True))
        for n, data in initial_nodes:
            current_reasoning = data['current_reasoning'] # List of dictionaries    
            data['number_steps'] = len(current_reasoning)
            data['hover'] = DesignReasoningVisualizer._get_node_hover_text(data)
            if 'score' in data:
                if data['score'] > 8:
                    data['color'] = 'lightgreen'
                elif data['score'] >= 6:
                    data['color'] = 'orange'
            if 'is_done' in data and data['is_done'] == True: # This is a truly terminal node
                data['color'] = 'green'
            if copy_graph.in_degree(n) == 0: #it's the root
                data['color'] = 'black'
        
        for n, data in initial_nodes:
            if expand_reasoning: # and (graph.out_degree(n) == 0): # It's a leaf node:
                current_reasoning = data['current_reasoning'] # List of dictionaries  
                # print("Processing", len(current_reasoning), "reasoning steps for node", n)
                _ = DesignReasoningVisualizer._get_internal_nodes(copy_graph, n, current_reasoning, filter_history=filter_history)

        return copy_graph

    @staticmethod
    def _get_external_nodes(graph, internal_node):
        pred_external = None
        sucs_external = []
        
        preds = list(graph.predecessors(internal_node)) # It's either 1 node or an empty list
        if len(preds) > 0:
            root = preds[0]
            if not graph.nodes[root]['internal']:
                pred_pred_external = list(graph.predecessors(root))
                pred_external = None if (len(pred_pred_external) == 0) else pred_pred_external[0]
        
        sucs = list(graph.successors(internal_node))  # It's either 1 node or an empty list
        if len(sucs) == 0:
            preds = list(graph.predecessors(internal_node))
            root = None
            while len(preds) > 0:
                root = preds[0]
                if not graph.nodes[root]['internal']:
                    break
                preds = list(graph.predecessors(root))
            if root is not None:
                sucs = list(graph.successors(root))
                sucs_external = [s for s in sucs if not graph.nodes[s]['internal']]

        return pred_external, sucs_external

    @staticmethod
    def remove_step_nodes(graph: DiGraph) -> DiGraph:
        external_nodes_to_remove = []
        n_preds_dict = {}
        n_sucs_dict = {}
        for n, data in graph.nodes(data=True):
            if not data['internal']:
                external_nodes_to_remove.append(n)
            else:
                pred_external, sucs_external = DesignReasoningVisualizer._get_external_nodes(graph, n)
                # print(n, pred_external, sucs_external)
                
                # Get the last internal node from the predecessor
                pred_internal = None
                if pred_external is not None:
                    for s in graph.successors(pred_external):
                        if graph.nodes[s]['internal']:
                            pred_internal = s
                            break
                    if pred_internal is not None:
                        next_internals = list(graph.successors(pred_internal))
                        while len(next_internals) > 0: # Iterate until the end of sequence of internal nodes
                            pred_internal = next_internals[0]
                            next_internals = list(graph.successors(pred_internal))
                # print("\tpreds internal:", pred_internal)
                n_preds_dict[n] = pred_internal
                
                # Get the first internal node from each successor
                sucs_internal = []
                for s in sucs_external:
                    suc_internal = None
                    for p in graph.successors(s):
                        if graph.nodes[p]['internal']:
                            suc_internal = p
                            break
                    if suc_internal is not None:
                        sucs_internal.append(suc_internal)
                # print("\tsucs internal:", sucs_internal)
                n_sucs_dict[n] = sucs_internal
        
        # Update the graph
        for k, v in n_preds_dict.items():
            if v is not None:
                graph.add_edge(v, k)
        for k, v in n_sucs_dict.items():
            for s in v:
                graph.add_edge(k, s)
        graph.remove_nodes_from(external_nodes_to_remove)
        return graph

    def show(self, expand_reasoning=True, filter_history=True, hide_nodes=False):
        new_graph = DesignReasoningVisualizer._decorate_graph(self.graph, expand_reasoning=expand_reasoning, filter_history=filter_history)
        if hide_nodes:
            new_graph = DesignReasoningVisualizer.remove_step_nodes(new_graph)
        new_graph, _ = DesignReasoningVisualizer._add_graph_layout(new_graph, invert=-1)
        return gv.d3(new_graph)