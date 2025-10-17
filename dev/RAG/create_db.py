import os
from KnowledgeBuilder import KnowledgeBuilder
qwen_apiKey = ""
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey
builder = KnowledgeBuilder()
file_list = [
    "lustre_v6_ref_man.pdf",
    "kind2_v2.0.0.pdf"
]
lib = builder.build_library(file_list)
