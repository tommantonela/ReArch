# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from typing import List
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings as chroma_settings

from llama_index.llms.openai import OpenAI
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryResult
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core import SimpleDirectoryReader, PropertyGraphIndex
from llama_index.core.graph_stores.types import EntityNode, ChunkNode, Relation
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from llama_index.core.base.response.schema import Response

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore

from sentence_transformers import util

import os

import networkx as nx
from pydot import Dot, Node, Edge

from tqdm import tqdm
import json


class DesignStore:

    CHROMADB_PATH = "./system_chromadb"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 256
    CHUNK_OVERLAP = 10

    def __init__(self, system:str=None, requirements:dict=None, qualities:dict=None,
                 create=False, path=None) -> None:

        client_path = path if path is not None else self.CHROMADB_PATH

        # print(Settings.embed_model)
        model_name = Settings.embed_model.model_name if Settings.embed_model is not None else self.EMBEDDINGS_MODEL
        print("model name:", model_name)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.persistent_client = chromadb.PersistentClient(path=client_path, settings=chroma_settings(allow_reset=True))

        self.embed_model = HuggingFaceEmbedding(model_name=model_name)

        print(create)
        if create:
            print("Deleting existing collections ...")
            self.persistent_client.reset()
            self.system_collection = self.persistent_client.get_or_create_collection("system", embedding_function=self.sentence_transformer_ef)
            self.requirements_collection = self.persistent_client.get_or_create_collection("requirements", embedding_function=self.sentence_transformer_ef)
            self.decisions_collection = self.persistent_client.get_or_create_collection("decisions", embedding_function=self.sentence_transformer_ef)
            self.qualities_collection = self.persistent_client.get_or_create_collection("qualities", embedding_function=self.sentence_transformer_ef)

            # Add the documents here
            print("Creating collections and adding data ...")
            if system is not None:
                sys_id = system['id']
                sys_description = system['description']
                # print(sys_description)
                # parser = SentenceSplitter(chunk_size=DesignStore.CHUNK_SIZE, chunk_overlap=DesignStore.CHUNK_OVERLAP)
                system_node = TextNode(text=sys_description)
                nodes = [system_node] # parser.get_nodes_from_documents([Document(text=sys_description)])
                print("**System:", sys_id, len(nodes))
                # for n in nodes:
                #     print(n)
                #     # print(n.id_, n.text, n.node_id)
                all_ids = [n.node_id for n in nodes]
                all_docs = [n.text for n in nodes]
                all_metadatas = [{"source": sys_id} for _ in nodes]
                self.system_collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
                print("  check:", self.system_collection.count())
            
            if requirements is not None:
                print("**Functional requirements:", len(requirements))
                # print(requirements)
                # pprint.pprint(requirements)
                all_ids = [r['id'] for r in requirements]
                all_docs = [r['description'] for r in requirements]
                
                # Store the patterns for each requirement
                all_patterns = [r['patterns'] for r in requirements]
                all_metadatas = DesignStore._convert_list_to_metadata(all_patterns, prefix="pattern")
                # print(all_metadatas)
                if len(all_ids) > 0:
                    self.requirements_collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
                print("  check:", self.requirements_collection.count())
            
            if qualities is not None:
                print("**Quality-attribute requirements:", len(qualities))
                # print(qualities)
                # pprint.pprint(qualities)
                all_ids = [r['id'] for r in qualities]
                all_docs = [r['description'] for r in qualities]
                
                # Store the requirements for each quality
                all_requirements = [r['requirements'] for r in qualities]
                all_metadatas_requirements = DesignStore._convert_list_to_metadata(all_requirements, prefix="requirement")
                # print(all_metadatas_requirements)
                all_quality_attributes = [r['qualities'] for r in qualities]
                all_metadatas_qualities = DesignStore._convert_list_to_metadata(all_quality_attributes, prefix="quality")
                # print(all_metadatas_qualities)
                all_metadatas = []
                for r,qa in zip(all_metadatas_requirements, all_metadatas_qualities):
                    all_metadatas.append(r | qa)
                print(all_metadatas)
                if len(all_ids) > 0:
                    self.qualities_collection.add(ids=all_ids, documents=all_docs, metadatas=all_metadatas)
                print("  check:", self.qualities_collection.count())

        else:
            print("Recovering existing collections ...")
            self.system_collection = self.persistent_client.get_or_create_collection("system", embedding_function=self.sentence_transformer_ef)
            print("**System:", self.system_collection.count())
            self.requirements_collection = self.persistent_client.get_or_create_collection("requirements", embedding_function=self.sentence_transformer_ef)
            print("**Requirements:", self.requirements_collection.count())
            self.quality_collection = self.persistent_client.get_or_create_collection("qualities", embedding_function=self.sentence_transformer_ef)
            print("**Quality-attribute scenarios:", self.quality_collection.count())
            self.decisions_collection = self.persistent_client.get_or_create_collection("decisions", embedding_function=self.sentence_transformer_ef)
            print("**Decisions:", self.decisions_collection.count())

    @staticmethod
    def _convert_list_to_metadata(all_items:List, prefix):
        all_metadatas = []
        for dp_list in all_items:
            md = dict()
            dp_list = list(set(dp_list))
            for i, x in enumerate(dp_list):
                    md[prefix+"_"+str(i+1)] = x
            if len(md) > 0:
                all_metadatas.append(md)
            else:
                all_metadatas.append(None)
        return all_metadatas

    def get_system(self, id:str=None) -> TextNode:
        node = None
        if id is None: 
            result = self.system_collection.get()
            if len(result['ids']) > 0: # Assemble all the pieces of the system description (there can be overlapping chunks)
                system_id = result['metadatas'][0]['source']
                system_text = '\n'.join(result['documents'])
                node = TextNode(id_=system_id, text=system_text)
        else:
            result = self.system_collection.get(ids=[id])
            if len(result['ids']) > 0:
                node = TextNode(id_=result['ids'][0], text=result['documents'][0], metadata=result['metadatas'][0])
        return node
    
    def get_requirement(self, id:str) -> TextNode:
        result = self.requirements_collection.get(ids=[id])
        node = None
        if len(result['ids']) > 0: # No embeddings returned
            if result['metadatas'][0] is not None:
                node = TextNode(id_=result['ids'][0], text=result['documents'][0], metadata=result['metadatas'][0])
            else:
                node = TextNode(id_=result['ids'][0], text=result['documents'][0])
        return node
    
    def get_patterns_for_requirement(self, id:str) -> List[TextNode]:
        result = self.requirements_collection.get(ids=[id])
        patterns = []
        if (len(result['ids']) > 0) and (result['metadatas'][0] is not None):
            patterns = [TextNode(text=p) for k,p in result['metadatas'][0].items() if "pattern" in k]
        return patterns
    
    def get_requirements(self, include_embeddings=False) -> List[TextNode]:
        if include_embeddings:
            result = self.requirements_collection.get(include=['embeddings', 'documents', 'metadatas'])
        else:
            result = self.requirements_collection.get()
        requirements = []
        if len(result['ids']) > 0:
            for idx, id in enumerate(result['ids']):
                doc = result['documents'][idx]
                metad = result['metadatas'][idx]
                if metad is not None:
                    if include_embeddings:
                        metad['embedding'] = result['embeddings'][0]
                    requirements.append(TextNode(id_=id, text=doc, metadata=metad))
                else: # In case metadata is void
                    requirements.append(TextNode(id_=id, text=doc))
        return requirements
    
    def get_quality_attribute_scenarios(self, include_embeddings=False) -> List[TextNode]:
        if include_embeddings:
            result = self.qualities_collection.get(include=['embeddings', 'documents', 'metadatas'])
        else:
            result = self.qualities_collection.get()
        qualities = []
        if len(result['ids']) > 0:
            for idx, id in enumerate(result['ids']):
                doc = result['documents'][idx]
                metad = result['metadatas'][idx]
                if include_embeddings:
                    metad['embedding'] = result['embeddings'][0]
                qualities.append(TextNode(id_=id, text=doc, metadata=metad))
        return qualities

    def get_decision(self, id:str) -> TextNode:
        result = self.decisions_collection.get(ids=[id])
        node = None
        if len(result['ids']) > 0: # No embeddings returned
            node = TextNode(id_=result['ids'][0], text=result['documents'][0], metadata=result['metadatas'][0])
        return node

    def get_decisions(self, include_embeddings=False) -> List[TextNode]:
        if include_embeddings:
            result = self.decisions_collection.get(include=['embeddings', 'documents', 'metadatas'])
        else:
            result = self.decisions_collection.get()
        decisions = []
        if len(result['ids']) > 0:
            for idx, id in enumerate(result['ids']):
                doc = result['documents'][idx]
                metad = result['metadatas'][idx]
                if include_embeddings:
                    metad['embedding'] = result['embeddings'][0]
                decisions.append(TextNode(id_=id, text=doc, metadata=metad))
        return decisions

    def get_requirements_for_decision(self, id:str) -> List[TextNode]:
        result = self.decisions_collection.get(ids=[id]) # No embeddings returned
        requirements = [TextNode(id_=req, text=id) for k,req in result['metadatas'][0].items() if "requirement" in k]
        return requirements
    
    def add_requirement(self, id:str, description:str, candidate_patterns:List=None, update=False):
        # If the id already exists, it should override it in the collection
        result = self.requirements_collection.get(ids=[id])
        if len(result['ids']) > 0:
            print("Warning: requirement already exists, overwriting ...", id, result)
            update = True

        all_metadatas = None
        if candidate_patterns is not None:
            all_metadatas = DesignStore._convert_list_to_metadata([candidate_patterns], prefix="pattern")
        # print(candidate_patterns, all_metadatas)

        if not update:
            if all_metadatas is not None:
                self.requirements_collection.add(ids=[id], documents=[description], metadatas=all_metadatas)
            else: 
                self.requirements_collection.add(ids=[id], documents=[description])
        else:
            if all_metadatas is not None:
                self.requirements_collection.update(ids=[id], documents=[description], metadatas=all_metadatas)
            else:
                self.requirements_collection.update(ids=[id], documents=[description])

    def set_candidate_patterns_for_requirement(self, id:str, patterns:list):
        result = self.requirements_collection.get(ids=[id])
        if len(result['ids']) == 0:
            print("Error: requirement not found, cannot add patterns ...", id, result)
            return
        self.add_requirement(result['ids'][0], result['documents'][0], patterns, update=True)

    def add_decision(self, id:str, decision:str, requirements:List=None, main_pattern=None, update=False):
        # If the id already exists, it should override it in the collection
        result = self.decisions_collection.get(ids=[id])
        if len(result['ids']) > 0:
            print("Warning: decision already exists, overwriting ...", id, result)
            update = True

        all_metadatas = None
        if requirements is not None:
            all_metadatas = DesignStore._convert_list_to_metadata([requirements], prefix="requirement")
        if main_pattern is not None:
            if all_metadatas is None:
                all_metadatas = []
                all_metadatas.append({"pattern": main_pattern})
            else:
                all_metadatas[0]["pattern"] = main_pattern
        # print(requirements, all_metadatas)

        if not update:
            if all_metadatas is not None:
                self.decisions_collection.add(ids=[id], documents=[decision], metadatas=all_metadatas)
            else:
                self.decisions_collection.add(ids=[id], documents=[decision])
        else:
            if all_metadatas is not None:
                self.decisions_collection.update(ids=[id], documents=[decision], metadatas=all_metadatas)
            else:
                self.decisions_collection.update(ids=[id], documents=[decision])
    
    def set_requirements_for_decision(self, id:str, requirements:list):
        result = self.decisions_collection.get(ids=[id])
        if len(result['ids']) == 0:
            print("Error: decision not found, cannot add requirements ...", id, result)
            return
        self.add_decision(result['ids'][0], result['documents'][0], requirements, update=True)
    
    def search_related_decisions(self, decision:str, k:int=3, threshold=0.5, semantic_search=False) -> List[NodeWithScore]:
        decisions = [] # dict()
        target_decision = decision
        if not semantic_search:
            result = self.decisions_collection.get(ids=[target_decision])
            if len(result['ids']) > 0:
                target_decision = result['documents'][0]
            else:
                return decisions
        # Semantic search
        decs = self.decisions_collection.query(query_texts=[target_decision], n_results=k) 
        # print(decs)
        for d, distance, description, md in zip(decs['ids'][0], decs['distances'][0], decs['documents'][0], decs['metadatas'][0]):
            if (target_decision != description) and (distance <= threshold) and (distance > 0.0):
                # decisions[d] = description
                n = TextNode(id_=d, text=description, metadata=md)
                decisions.append(NodeWithScore(node=n, score=distance))
        return decisions

    def find_decision(self, description:str, k:int=1, threshold=0.5) -> List[NodeWithScore]:
        decisions = [] # dict()
        decs = self.decisions_collection.query(query_texts=[description], n_results=k)
        for d, distance, description, md in zip(decs['ids'][0], decs['distances'][0], decs['documents'][0], decs['metadatas'][0]):
            if distance <= threshold:
                #decisions[d] = description
                n = TextNode(id_=d, text=description, metadata=md) # No embeddings returned
                decisions.append(NodeWithScore(node=n, score=distance))
        return decisions
    
    def search_decisions_for_requirement(self, requirement: str, k:int=3, semantic_search=False, 
                                         threshold=0.5, forbidden_decision=None) -> List[TextNode]:
        decisions = [] # dict()
        if not semantic_search: # requirement is considered as an id
            nodes = self.get_decisions()
            # for d, metadata, description in zip(docs['ids'], docs['metadatas'], docs['documents']):
            for n in nodes:
                d = n.node_id
                metadata = n.metadata
                description = n.text 
                for k, v in metadata.items():
                    if ("requirement" in k) and (requirement == v): 
                        # decisions[d] = description
                        n = TextNode(id_=d, text=description, metadata=metadata)
                        decisions.append(n)
            return decisions
        else: # requirement is used for semantic search
            reqs = self.requirements_collection.query(query_texts=[requirement], n_results=k)
            # print(reqs)
            emb = None
            if forbidden_decision is not None:
                emb = self.sentence_transformer_ef([forbidden_decision])[0]
            for r, distance in zip(reqs['ids'][0], reqs['distances'][0]):
                if (distance <= threshold) and (distance > 0.0):
                    # print("Checking", r, distance)
                    decisions_for_r = self.search_decisions_for_requirement(r, k, semantic_search=False)
                    if emb is not None:
                        # temp = list(decisions_for_r.keys())
                        old = [d.id_ for d in decisions_for_r]
                        # decisions_for_r = {k: v for k, v in decisions_for_r.items() if util.cos_sim(emb, self.sentence_transformer_ef([v])[0]) < 1.0}
                        decisions_for_r = [n for n in decisions_for_r if util.cos_sim(emb, self.sentence_transformer_ef([n.text])[0]) < 1.0]
                        print("Filtering forbidden decisions ...", len(old), len(decisions_for_r))
                    decisions.update(decisions_for_r)
            return decisions 
    
    def search(self, query:str, collection:str, k:int=3) -> List[NodeWithScore]:
        if not (collection in ["requirements", "decisions", "system", "qualities"]):
           print("Error: collection not found ...", collection)
           return []
        if collection == "system":
            chroma_collection = self.system_collection
        elif collection == "requirements":
            chroma_collection = self.requirements_collection
        elif collection == "qualities":
            chroma_collection = self.qualities_collection
        else:
            chroma_collection = self.decisions_collection

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        print("There are", vector_store._collection.count(), "items in the", collection, "collection")

        query_embedding = self.embed_model.get_query_embedding(query)
        vectorstore_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=k)
        result = vector_store.query(vectorstore_query) # It relies on LlamaIndex wrapper
        nodes = []
        for idx, node, score in zip(result.ids, result.nodes, result.similarities):
            n = TextNode(id_=idx, text=node.text, metadata=node.metadata) # No embeddings returned
            nodes.append(NodeWithScore(node=n, score=score))
        return nodes

    def get_named_patterns(self) -> List[str]:
        result = self.requirements_collection.get(include=['metadatas'])
        patterns = []
        for md in result['metadatas']:
            if md is not None:
                patterns.extend([p for k,p in md.items() if "pattern" in k])
        result = self.decisions_collection.get(include=['metadatas'])
        for md in result['metadatas']:
            if md is not None:
                patterns.extend([p for k,p in md.items() if "pattern" in k])
        return list(set(patterns))
    
    # TODO: Review and simplify the logic of this method
    def to_networkx(self, graph_id, requirements_only=False, include_embeddings=False) -> nx.DiGraph:
        graph = nx.DiGraph(id=graph_id, label=graph_id)

        # Add all the requirements
        patterns = set()
        requirements = self.get_requirements(include_embeddings=include_embeddings)
        for req in requirements:
            if include_embeddings:
                graph.add_node(req.node_id, text=req.text, kind="REQUIREMENT", embedding=req.metadata['embedding'], shape='circle', color='blue')
            else:
                graph.add_node(req.node_id, text=req.text, kind="REQUIREMENT", shape='circle', color='blue')
            for p_node in self.get_patterns_for_requirement(req.node_id):
                patterns.add(p_node.text) # All the patterns for this requirement
    
        # Link parent-child requirements (refinement)
        for rchild in requirements:
            for rparent in requirements:
                prefix = rchild.node_id.split('.')
                if (len(prefix) == 2) and (prefix[0] == rparent.node_id): # it`s a child
                    # print("--Adding edge", rparent.node_id, rchild.node_id)
                    graph.add_edge(rparent.node_id, rchild.node_id, kind='IS_REFINED_BY', color='blue')

        if requirements_only:
            return graph
        
        # Add all the decisions
        decisions = self.get_decisions()
        for dec in self.get_decisions(include_embeddings=include_embeddings):
            if include_embeddings:
                graph.add_node(dec.node_id, text=dec.text, kind="DECISION", embedding=dec.metadata['embedding'], shape='diamond', color='red')
            else:
                graph.add_node(dec.node_id, text=dec.text, kind="DECISION", shape='diamond', color='red')
            p = dec.metadata['pattern']
            patterns.add(p) # The pattern for this decision

        pattern_embeddings = self.sentence_transformer_ef(list(patterns))
        patterns_dict = {k:v for k,v in zip(patterns, pattern_embeddings)}
        # print(json.dumps(patterns_dict, indent=4))
    
        # TODO: Perform entity disambiguation for patterns using embeddings

        # Add all the patterns
        for pat, e in patterns_dict.items():
            if include_embeddings:
                graph.add_node(pat, kind="PATTERN", shape='rectangle', fillcolor='gray', style='filled', embedding=e)
            else:
                graph.add_node(pat, kind="PATTERN", shape='rectangle', fillcolor='gray', style='filled')

        # Add links betweeen requirements and patterns (can-satisfy)
        for req in requirements:
            for p_node in self.get_patterns_for_requirement(req.node_id):
                graph.add_edge(p_node.text, req.node_id, kind='CAN_SATISFY', style='dashed')
        
        # Add links between decisions and patterns and requirements (implements, satisfies)
        for dec in decisions:
            reqs = self.get_requirements_for_decision(dec.node_id)
            for req in reqs:
                graph.add_edge(dec.node_id, req.node_id, kind='SATISFIES', color='red')
            p = dec.metadata['pattern']
            graph.add_edge(dec.node_id, p, kind='IMPLEMENTS', color='red', style='dashed')

        return graph
        # return nx.convert_node_labels_to_integers(graph)
    
    # https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/property_graph/graph_store.ipynb
    def as_graphstore(self, include_embeddings=False):

        patterns = set()
        requirements = []
        for req in self.get_requirements(include_embeddings=include_embeddings):
            if include_embeddings:
                entity = EntityNode(label='REQUIREMENT', name=req.id_, embedding=req.metadata['embedding'], properties={'text': req.text})
            else:
                entity = EntityNode(label='REQUIREMENT', name=req.id_, properties={'text': req.text})
            requirements.append(entity)
            for p_node in self.get_patterns_for_requirement(req.id_):
                patterns.add(p_node.text) # All the patterns for this requirement
        
        decisions = []
        implements = []
        for dec in self.get_decisions(include_embeddings=include_embeddings):
            # print(dec.metadata)
            entity = EntityNode(label='DECISION', name=dec.id_, embedding=dec.metadata['embedding'], properties={'text': dec.text})
            decisions.append(entity)
            p = dec.metadata['pattern']
            patterns.add(p) # The pattern for this decision
            rel = Relation(label="IMPLEMENTS", source_id=dec.id_, target_id=p)
            implements.append(rel)

        pattern_embeddings = self.sentence_transformer_ef(list(patterns))
        patterns_dict = {k:v for k,v in zip(patterns, pattern_embeddings)}

        patterns = []
        for p, e in patterns_dict.items():
            if include_embeddings:
                entity = EntityNode(label='PATTERN', name=p, embedding=e, properties={'text': p})
            else:
                entity = EntityNode(label='PATTERN', name=p, properties={'text': p})
            patterns.append(entity)

        can_satisfy = []
        for req in self.get_requirements():
            for p_node in self.get_patterns_for_requirement(req.id_):
                rel = Relation(label="CAN_SATISFY", source_id=p_node.text, target_id=req.id_) 
                can_satisfy.append(rel)
        
        is_refined_by = []        
        for rchild in requirements:
            for rparent in requirements:
                prefix = rchild.name.split('.')
                if (len(prefix) == 2) and (prefix[0] == rparent.name): # it`s a child
                    rel = Relation(label="IS_REFINED_BY", source_id=rparent.name, target_id=rchild.name) 
                    is_refined_by.append(rel)

        satisfies = []
        for dec in decisions:
            reqs = self.get_requirements_for_decision(dec.name)
            for r in reqs:
                rel = Relation(label="SATISFIES", source_id=dec.name, target_id=r.text)
                satisfies.append(rel)

        graph_store = SimplePropertyGraphStore()
        graph_store.upsert_nodes(requirements + decisions + patterns)
        graph_store.upsert_relations(can_satisfy + is_refined_by + implements + satisfies)

        index = PropertyGraphIndex(nodes=[], property_graph_store=graph_store, 
                                   llm=None, embed_kg_nodes=False, use_async = False,
                                   vector_store=None, 
                                   embed_model = HuggingFaceEmbedding(model_name=self.EMBEDDINGS_MODEL), #'local',
                                   kg_extractors=[],  show_progress=True)
        return graph_store, index


# TODO: Visualization of nodes using UMAP/BERTopic (zero-shot)?
# https://medium.com/@sandyshah1990/enhancing-document-retrieval-with-reranker-and-umap-visualization-exploring-llm-techniques-8911caf0cd43
# https://github.com/SandyShah/llama_index_experiments/blob/main/umap_reranker.ipynb
# Or maybe an EDA for RAG: https://itnext.io/visualize-your-rag-data-eda-for-retrieval-augmented-generation-0701ee98768f
# https://github.com/Renumics/spotlight
class KnowledgeStore:

    CHROMADB_PATH = "./patterns_chromadb"
    EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 20

    def __init__(self, create=False, path=None) -> None:

        self.db_path = path if path is not None else self.CHROMADB_PATH

        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.EMBEDDINGS_MODEL)
        self.persistent_client = chromadb.PersistentClient(path=self.db_path, settings=chroma_settings(allow_reset=True))

        # self.embeddings = SentenceTransformerEmbeddings(model_name=KnowledgeStore.EMBEDDINGS_MODEL) # Local embeddings
        self.embeddings = LangchainEmbedding(HuggingFaceEmbeddings(model_name="sentence-transformers/"+KnowledgeStore.EMBEDDINGS_MODEL))

        if create:
            print("Deleting existing collections ...")
            self.persistent_client.reset()
        
        self.dpatterns_collection = self.persistent_client.get_or_create_collection("design_patterns", embedding_function=self.sentence_transformer_ef)
        self.styles_collection = self.persistent_client.get_or_create_collection("architectural_styles", embedding_function=self.sentence_transformer_ef)
        self.microservices_collection = self.persistent_client.get_or_create_collection("microservice_patterns", embedding_function=self.sentence_transformer_ef)

        self.dpatterns_vectordb = ChromaVectorStore(chroma_collection=self.dpatterns_collection)  
        print("There are", self.dpatterns_vectordb._collection.count(), "chunks in the design patterns collection")
        self.dp_index = VectorStoreIndex.from_vector_store(self.dpatterns_vectordb, embed_model=self.embeddings)

        self.styles_vectordb = ChromaVectorStore(chroma_collection=self.styles_collection)  
        print("There are", self.styles_vectordb._collection.count(), "chunks in the architectural styles collection")
        self.styles_index = VectorStoreIndex.from_vector_store(self.styles_vectordb, embed_model=self.embeddings)

        self.microservices_vectordb = ChromaVectorStore(chroma_collection=self.microservices_collection)  
        print("There are", self.microservices_vectordb._collection.count(), "chunks in the microservice patterns collection")
        self.microservices_index = VectorStoreIndex.from_vector_store(self.microservices_vectordb, embed_model=self.embeddings)
    
    def _parse_nodes(self, directory:str):
        reader = SimpleDirectoryReader(input_dir=directory, required_exts=['.pdf'])
        documents = reader.load_data()
        print("Ingesting PDFs ...", directory, len(documents), "documents")
        
        # TODO: Try a semantic chunker as an alternative
        # https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking/
        parser = SentenceSplitter(chunk_size=KnowledgeStore.CHUNK_SIZE, chunk_overlap=KnowledgeStore.CHUNK_OVERLAP)
        nodes = parser.get_nodes_from_documents(documents)
        print(len(nodes), "nodes")
        return nodes
    
    def ingest_architectural_patterns(self, directory:str):
        nodes = self._parse_nodes(directory)
        # Ingest the database and create the index
        storage_context = StorageContext.from_defaults(vector_store=self.styles_vectordb)
        self.styles_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embeddings)
        # self.styles_index.storage_context.persist(persist_dir=self.db_path+"\indexes\design-patterns")
  
    def ingest_design_patterns(self, directory:str):
        nodes = self._parse_nodes(directory)
        # Ingest the database and create the index
        storage_context = StorageContext.from_defaults(vector_store=self.dpatterns_vectordb)
        self.dps_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embeddings)
        # self.dps_index.storage_context.persist(persist_dir=self.db_path+"\indexes\design-patterns")

    def ingest_microservice_patterns(self, directory:str):
        nodes = self._parse_nodes(directory)
        # Ingest the database and create the index
        storage_context = StorageContext.from_defaults(vector_store=self.microservices_vectordb)
        self.microservices_index = VectorStoreIndex(nodes, storage_context=storage_context, embed_model=self.embeddings)
        # self.microservices_index.storage_context.persist(persist_dir=self.db_path+"\indexes\design-patterns")
    
    def search(self, query:str, collection:str, k:int=3) -> List[NodeWithScore]:
        if collection == "architectural_styles":
            vectordb = self.styles_vectordb
        elif collection == "design_patterns":
            vectordb = self.dpatterns_vectordb
        elif collection == "microservice_patterns":
            vectordb = self.microservices_vectordb
        else:
            print("Error: collection not found ...", collection)
            return []

        # docs = vectordb.similarity_search(query, k=k) # It relies on Langchain wrapper
        embedding = self.embeddings._get_query_embedding(query)
        vectorstore_query = VectorStoreQuery(query_embedding=embedding, similarity_top_k=k)
        result = vectordb.query(vectorstore_query) # It relies on LlamaIndex wrapper
        nodes = []
        for idx, node, score in zip(result.ids, result.nodes, result.similarities):
            n = TextNode(id_=idx, text=node.text, metadata=node.metadata)
            nodes.append(NodeWithScore(node=n, score=score))
        return nodes
    
    def _get_database(self, collection:str):
        vectordb = None
        if collection == "architectural_styles":
            vectordb = self.styles_vectordb
        elif collection == "design_patterns":
            vectordb = self.dpatterns_vectordb
        elif collection == "microservice_patterns":
            vectordb = self.microservices_vectordb
        else:
            print("Error: collection not found ...", collection)
            return None
        
        return vectordb
    
    # TODO: Implement LlamaIndex retrievers (or query engine directly?)
    # https://docs.llamaindex.ai/en/stable/examples/retrievers/simple_fusion/
    def get_retriever(self, collection:str=None,  k:int=3, num_queries=1, threshold:float=0.5):
                            # llm=None, synthesize_response=True, num_queries=1):
        if collection is None:
            print("Warning: collection not found ...", collection)
            return None
        
        if collection == 'all':
            print("Creating LOTR ...")
            retrievers = []
            all_collections = ["design_patterns", "microservice_patterns", "architectural_styles"]
            vector_stores = [self._get_database(collection=c) for c in all_collections]
            retrievers = [VectorStoreIndex.from_vector_store(vs).as_retriever() for vs in vector_stores]
            lotr_retriever = QueryFusionRetriever(
                retrievers,
                similarity_top_k=k,
                num_queries=num_queries,  # set this to 1 to disable query generation
                use_async=False,
                verbose=False
                # query_gen_prompt="...",  # we could override the query generation prompt here
            )
            # query_engine = RetrieverQueryEngine.from_args(lotr_retriever)
            return lotr_retriever
        else:
            chroma_db = self._get_database(collection=collection)
            index = VectorStoreIndex.from_vector_store(chroma_db)
            retriever = index.as_retriever(similarity_top_k=k)
            # llm_model = llm if llm is not None else Settings.llm
            # query_engine = index.as_query_engine(llm=llm_model, similarity_top_k=k, synthesize_response=synthesize_response)
            return retriever

    PATTERN_DESCRIPTION_TEMPLATE = (
        "You are an expert software architect. Please provide a concise description of {pattern}, including its usage in software design."
    ) 
       
    def get_pattern_descriptions(self, patterns:List[str], collection:str='all') -> List[Response]:
        query_engine = self.get_query_engine(collection=collection)
        qa_template = PromptTemplate(KnowledgeStore.PATTERN_DESCRIPTION_TEMPLATE)
        # you can create text prompt (for completion API)
        descriptions = []
        for p in tqdm(patterns):
            pattern_str = p
            if not pattern_str.lower().endswith('pattern'):
                pattern_str = " pattern"
            prompt = qa_template.format(pattern=pattern_str)
            response = query_engine.query(prompt)
            descriptions.append(response)

        return descriptions
