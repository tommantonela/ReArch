# ReArch
Reproducibility kit for the paper *"Architecture Exploration and Reflection using LLM-based Agents"*

The exploration of architecture alternatives is an essential part of the architecture design process, in which designers search and assess solutions for their requirements. Although automated tools and techniques have been proposed for this process, their reliance on formal specifications (e.g., quality-attribute analyses, design decisions) introduces adoption barriers. Nowadays, the emergence of generative AI techniques creates an opportunity for leveraging natural language representations in architecture design, particularly through LLM-based agents. To date, these agents have been mostly focused on coding-related tasks or requirements analysis. In this work, we investigate an approach for defining design agents powered by an LLM, which can autonomously search for architectural patterns and tactics for a particular system and requirements using a textual format. In addition to incorporating architectural knowledge, these agents can reflect on the pros and cons of the proposed decisions, enabling a feedback loop towards improving the quality of the decisions. We present a proof-of-concept called **ReArch** that adapts elements from the *ReAct* and *LATS* agent frameworks, and discuss initial results of applying our design agents to an architectural case study considering different patterns.

----
The artifacts provided include:
* The Python code that implements the **ReArch** agents using [LlamaIndex](https://www.llamaindex.ai/). The prompts used by the agents are embedded in their code.
* A vector database (Chroma) for the agents' tools to retrieve architectural knowlegdge, based on a RAG schema.
* A Python script for exercising the agents on all the requirements of the [CampusBike](https://github.com/shamimaaktar1/ChatGPT4SA) case study. This script runs in a batch mode and generates the different trajectories as JSON and HTML files.
* Example *Jupyter* notebooks for instantiating and running the **ReArch** agents on a particular requirement (and system context), and also for displaying the trajectories (JSON files) as graphs.
