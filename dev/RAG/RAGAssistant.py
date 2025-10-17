from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.llms import Tongyi
from langchain.prompts import PromptTemplate

def get_agent(name, temperature):
    if name == "gpt":
        return ChatOpenAI(model="gpt-4-0125-preview", temperature=temperature)
    elif name == "qwen":
        return Tongyi(model_name="qwen2.5-coder-32b-instruct", temperature=temperature)


class RAGAssistant:
    def __init__(self, vector_db, template, temperature=0.2, agent_name="qwen"):
        self.llm = get_agent(agent_name, temperature)
        self.retriever = vector_db.as_retriever(
            search_type="mmr",  # 最大边际相关性
            search_kwargs={"k": 5} 
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": self._build_prompt(template),
                "document_separator": "\n\n------\n"
            }
        )
    
    default_template = """你是一名Lustre语言专家，并可以使用kind2形式化验证工具，请严格根据以下知识片段回答问题：{context}

        问题：{question}
        回答要求：
        1. 如果涉及代码，必须生成可编译的完整示例，注意你的回答只需要给出代码，不需要给出任何多余信息或者分点格式
        2. 如果需求中没有验证规则，则一定不需要额外生成验证规则"""
    def _build_prompt(self, template = default_template):
        return PromptTemplate.from_template(template)

    def query(self, question):
        """执行RAG查询"""
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": [doc.metadata["source"] for doc in result["source_documents"]],
            "content": [doc.page_content for doc in result["source_documents"]]
        }