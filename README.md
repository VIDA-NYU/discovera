# 🧬 Gene Set Function Discovery with LLM-based Agents and Knowledge Retrieval

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue)](https://www.python.org/)
[![LLM-Enhanced](https://img.shields.io/badge/Powered%20by-LLMs-blueviolet)]()

This tool is an interactive, agent-based system that integrates traditional bioinformatics tools with large language models (LLMs) and retrieval-augmented generation (RAG) to support hypothesis generation and mechanistic discovery in functional genomics.

## 🔍 What It Does

Bridges the gap between computational analysis and interpretability in biomedical research. It is designed to assist researchers—regardless of their coding expertise—in:

- Interactively exploring gene sets associated with complex phenotypes  
- Conducting functional enrichment analyses  
- Summarizing mechanistic hypotheses using evidence from literature  
- Formulating data-grounded biological insights  

## ✨ Key Features

- ⚙️ **Modular System**: Combines established tools (e.g., GSEApy, INDRA) with custom modules for extensibility.  
- 📖 **LLM Integration**: Uses LLMs for natural language reasoning and explanation generation.  
- 🔎 **Retrieval-Augmented Generation**: Grounds summaries in real literature to improve accuracy and transparency.  
- 💬 **Chat Interface**: Enables intuitive, dialogue-based exploration of hypotheses and gene set functions.  

## 🧪 Use Case Example

In our initial deployment, this agent was used in the context of **endometrial carcinoma (EC)** to:

- Analyze gene sets linked to phenotypic features  
- Perform enrichment analysis on the resulting sets  
- Summarize literature-supported mechanisms of action  

## 🛠️ System Architecture

![Agent Architecture](images/discovera.pdf)

The system consists of:
- Enrichment tools (e.g., GSEApy)  
- INDRA for biological statement synthesis  
- LLM-enabled prompt orchestration  
- Chat-based user interface  

## 🚀 Getting Started

### Prerequisites

- Python 3.8+  
- [GSEApy](https://gseapy.readthedocs.io/)  
- [INDRA](https://indra.readthedocs.io/)  
- OpenAI or similar LLM API access  
