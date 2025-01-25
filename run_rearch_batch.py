
import warnings

import os
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI

from stores import KnowledgeStore, DesignStore
from archmind import ArchmindAssistant, ArchmindDesignAgent

import pprint
import pandas as pd

warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ENV_PATH = Path('.') / '<your-env-file>.env'
result = load_dotenv(dotenv_path=ENV_PATH.resolve(), override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini' 

Settings.embed_model = HuggingFaceEmbedding(model_name=DesignStore.EMBEDDINGS_MODEL)
llm = OpenAI(model=os.getenv('OPENAI_MODEL_NAME'), temperature=0.0)

# Load the design/knowledge store
chroma_db = "./patterns_chromadb" 
kstore = KnowledgeStore(path=chroma_db)
sys_id = "campus-bike" # "pharmacy-food" #"campus-bike" #"das-p1-2023"
chroma_db = "./"+sys_id+"_chromadb"
print("Accesing design store ...", chroma_db)
dstore = DesignStore(path=chroma_db)
print()

# Create and configure the assistant
assistant = ArchmindAssistant(dstore, kstore, llm, sys_id=sys_id, rag=True)
assistant.configure_retriever(collection="all", threshold=0.8)

design_agent = ArchmindDesignAgent(assistant, llm=llm)
assistant.clear_decisions()
decision_1 = "The system should be implemented applying object-oriented design constructs." # Furthermore, the system should adhere to a microservices architecture."
assistant.add_decision("D1", decision_1)
pprint.pprint(assistant.fetch_context(with_decisions=True))

print()
system_context = assistant.fetch_context()
print(system_context)
print()
# pprint.pprint(system_context)

for r in dstore.get_requirements(): # List the requirements of the system
    # print(r.id_, ":", r.text)
    msg = r.id_+ ": "+ r.text
    print(msg)

print()

####################################
root_path = "./runs/"
mode = "lats" # "react" "lats"

alternatives_filename = root_path+sys_id+"_alternatives_"+mode+".csv"
df_alternatives = pd.DataFrame()

n_cycles = 2 # For ReAct, For LATS is should be = 1
design_agent.create_agent(mode=mode, max_iterations=5, verbose=False) # ReAct
# design_agent.create_agent(mode=mode, num_expansions=3, max_rollouts=6, verbose=False) # LATS

for r in dstore.get_requirements(): 
    print("="*10, r.id_)
    requirement = r.text
    print(requirement)
    print()

    design_agent.reset_agent()
    alternatives = design_agent.run_agent(requirement, n_cycles=n_cycles)

    trajectory_filename = root_path+sys_id+"_"+r.id_.lower()+"_"+mode+".json"
    graph_output = design_agent.save_agent_state(filename=trajectory_filename, as_graph=True, alternatives_only=False)

    ll = []
    for idx, a in enumerate(alternatives):
        msg = str(idx+1)+". "+a
        # print(msg)
        pprint.pprint(msg)
        ll.append({'id':r.id_, 'requirement': requirement, 'num_alt':idx, 'alternative':a})
    
    df_alternatives = pd.concat([df_alternatives,pd.DataFrame(ll)])
    df_alternatives.to_csv(alternatives_filename,index=False)
    print()

print("="*10)
design_agent.reset_agent()
print("Done!")
print()