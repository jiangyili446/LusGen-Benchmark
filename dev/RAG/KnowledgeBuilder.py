import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
class KnowledgeBuilder:
    def __init__(self, chunk_size=512, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "§"]
        )
        self.embeddings = DashScopeEmbeddings(model="text-embedding-v3")
    def build_library(self, file_paths, save_path = "vector_store"):
        docs = []
        current_file = os.path.abspath(__file__)

        parent_dir = os.path.dirname(current_file)
        doc_dir = os.path.join(parent_dir, "docs")
        for path in file_paths:
            path = os.path.join(doc_dir, path)
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path)
            pages = loader.load()
            docs.extend(self.text_splitter.split_documents(pages))

        self.vector_db = FAISS.from_documents(docs, self.embeddings)
        target_dir = os.path.join(parent_dir, save_path)

        os.makedirs(target_dir, exist_ok=True)
        self.vector_db.save_local(target_dir)
        print(f"：{self.vector_db.index.d}")
        return len(docs)
    
if __name__ == "__main__":
    import os
    qwen_apiKey = ""
    os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey
    builder = KnowledgeBuilder()
    file_list = [
        "test.pdf"
    ]
    lib = builder.build_library(file_list)