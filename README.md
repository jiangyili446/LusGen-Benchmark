# LusGen: Leveraging LLMs for Safety-Critical Lustre Design and Requirements Traceability

## 1. Overview
Safety-critical systems require precise software modeling and requirements traceability, with Lustre widely used for formal design. However, large language models (LLMs) struggle with Lustre’s syntax and semantics. We propose LusGen, the first LLM-based Lustre generation approach that integrates domain knowledge via prompt engineering and retrieval-augmented generation, combined with syntax checking and semantic verification feedback for iterative refinement. LusGen also enables fine-grained requirements-to-design traceability compliant with DO-178C. We conduct an empirical study revealing hallucination patterns in Lustre generation to guide our approach. Evaluations on four datasets show LusGen improves syntax correctness to 93\%, semantic correctness to 62\%, and achieves over 85\% accuracy in traceability, demonstrating practical industrial applicability. We also provide a benchmark dataset to advance future research.
## 2. Directory Structure

```
LusGen
├── benchmark/          # Stores benchmark datasets for evaluating LLM-driven Lustre generation performance
│   ├── d1/             # Sub-directory 1 of benchmark datasets, containing 24 non-verification data from the Kind2 dataset
│   ├── d2/             # Sub-directory 2 of benchmark datasets, with 26 verification-included data from the Kind2 dataset
│   ├── d3/             # Sub-directory 3 of benchmark datasets, having 5 manually-annotated verification data from the VeCoGen experimental dataset
│   └── d4/             # Sub-directory 4 of benchmark datasets, consisting of 8 data from 1 real-world wind speed system case
└── dev/
│    ├── gen/    # Generates Lustre models
│    │    ├── prompt/    # Contains Prompt templates for different strategies
│    │    ├── veco_fv_v2.py    # Lustre generation with formal verification feedback
│    │    ├── lusgen_fv_v2.py    # Lustre generation with formal verification feedback + RAG
│    │    ├── llm_v2.py    # only llm generation
│    │    ├── veco_sc_v2.py    # Lustre generation with syntax check feedback
│    │    ├── lusgen_sc_v2.py    # Lustre generation with syntax check feedback + RAG
│    ├── RAG/    # Retrieval-Augmented Generation related components
│    │    ├── docs/    # Domain knowledge documents
│    │    ├── vector_store    # Knowledge vector database
│    │    └── KnowledgeBuilder.py    # Builds domain knowledge into vector database
│    ├── req_formal/    # Requirements formalization processing
│    │    ├── origin2formal.py    # Converts original requirements to formalized requirements
│    │    ├── req2xml.py    # Converts original requirements to XML
│    │    └── xml2nat.py    # Converts XML to formalized requirements
│    └── req_trace/    # Requirements tracing
│         └── req_trace.py    # Generates requirements traceability matrix
```
