# ReArch
Reproducibility kit for the paper *"Architecture Exploration and Reflection using LLM-based Agents"*, accepted at the NEMI track for ICSA 2025.

The exploration of architecture alternatives is an essential part of the architecture design process, in which designers search and assess solutions for their requirements. Although automated tools and techniques have been proposed for this process, their reliance on formal specifications (e.g., modeling languages) introduces adoption barriers. Nowadays, the emergence of generative AI techniques creates an opportunity for leveraging natural language representations in architecture design, particularly through LLM-based agents. To date, these agents have been mostly focused on coding-related tasks or requirements analysis. In this work, we investigate an approach for defining design agents, which can autonomously search for architectural patterns and tactics for a particular system and requirements using a textual format. In addition to incorporating architectural knowledge, these agents can reflect on the pros and cons of the proposed decisions, enabling a feedback loop towards improving the decisions’ quality. We present a proof-of-concept called **ReArch** that adapts elements from the [ReAct](https://arxiv.org/abs/2210.03629) and [LATS](https://arxiv.org/abs/2310.04406) agent frameworks, and discuss initial results of applying our LLM-based agents to a case study considering different patterns.

----
The artifacts provided include:
* The Python code that implements the **ReArch** agents using [LlamaIndex](https://www.llamaindex.ai/). The prompts used by the agents are embedded in their code.
* A vector database for the agents' tools to retrieve architectural knowlegdge using a [RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/) schema.
* A Python script for exercising the agents on all the requirements of the [CampusBike](https://github.com/shamimaaktar1/ChatGPT4SA) case study. This script runs in a batch mode and generates the different trajectories as JSON and HTML files.
* Example *Jupyter* notebooks for instantiating and running the **ReArch** agents on a particular requirement (and system context), and also for displaying the trajectories (JSON files) as graphs.

To run the design agents with the *[test_rearch.ipynb](https://github.com/tommantonela/ReArch/blob/main/test_rearch.ipynb)* or *[run_rearch_batch.ipynb](https://github.com/tommantonela/ReArch/blob/main/run_rearch_batch.py)* notebooks, a local Python environment needs to be configured. The experiments were performed with [OpenAI](https://openai.com/), thus an *OpenAI key* also needs to be provided.

The agents' tools relies on both software architecture and system-specific knowledge stored in [ChromaDB](https://www.trychroma.com/).

The *[view_trajectories.ipynb](https://github.com/tommantonela/ReArch/blob/main/view_trajectories.ipynb)* notebook is only for visualization purporses, so it assumes that the agents have been executed beforehand.

The **ReArch** prototype is based on the **[ArchMind](https://github.com/tommantonela/archmind)** project, which provides a chatbot for novice users to explore, analyze and document design decisions. Internally, **ArchMind** uses different prompts and database queries, which were encapsulated as tool for the **ReArch** agents. 

Our implementations of the design agents are based on the example implementations for [ReAct](https://docs.llamaindex.ai/en/stable/examples/agent/react_agent/) and [LATS](https://docs.llamaindex.ai/en/stable/examples/agent/lats_agent/) provided by LlamaIndex.


