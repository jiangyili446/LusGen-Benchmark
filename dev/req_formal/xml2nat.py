import os
root_dir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(root_dir)
from langchain_community.vectorstores import FAISS
from dev.RAG.RAGAssistant import RAGAssistant
from dev.RAG.KnowledgeBuilder import KnowledgeBuilder
from dev.const import gpt_apiKey, qwen_apiKey
os.environ["OPENAI_API_KEY"] = gpt_apiKey
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey

xml_template_raw = """
<Node name="First">
    <Inputs>
        <Param name="X" type="int"/>
    </Inputs>
    <Outputs>
        <Param name="Y" type="int"/>
    </Outputs>
    <Func>The initial value of Y is the value of input X, and the Y value of the previous moment is used at subsequent moments</Func>
</Node>
"""

nat_template_raw = """
Define the First node:
- Input: Accepts an integer input X
- Output: Returns an integer output Y
- Internal variables: None
- Functional logic: Y's initial value is the value of input X, and subsequent values are based on the previous value of Y.
- Validation rules: None.
"""

xml_template_formal ="""
<Node name="X">
    <Inputs>
      <Param name="i" type="bool"/>
    </Inputs>
    <Outputs>
      <Param name="x" type="int"/>
    </Outputs>
    <Vars>
      <Var name="OK" type="bool"/>
      <Const name="a" type="int" value="1"/>
    </Vars>
    <Func>The initial value of x is 0. If the previous value of x is less than 2, then the value of x is set to 1 minus the previous value of x. If the previous condition is not met and the input i is true, then the value of x is 3. In other cases, the value of x is set to 2.</Func>
    <Validations>
      <Rule expression="OK">x小于3</Rule>
    </Validations>
</Node>
"""

nat_template_formal = """
Define the X node:
- Input: Receives a Boolean input i
- Output: Returns an integer output x.
- Internal variables:
    - 1. OK: A Boolean internal variable used to record whether the previous condition is met.
    - 2. a: An integer constant with a value of 1.
- Functional logic:
    The initial value of x is 0. If the previous x is less than 2, then the value of x is set to 1 minus the previous x.
    If the previous condition is not met and input i is true, then the value of x is 3.
    In other cases, the value of x is set to 2.
- Validation rules:
    - OK: If x is less than 3, use the kind2 validation syntax to check whether it is always true.
"""

builder = KnowledgeBuilder()
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
save_path = "../RAG/vector_store"
target_dir = os.path.join(parent_dir, save_path)
vector_db = FAISS.load_local(target_dir, embeddings=builder.embeddings,
allow_dangerous_deserialization=True) 
template = """
You are a Lustre language expert and familiar with its Kind2 formal verification tool. Your task is to analyze user-entered requirements and output them in a natural, semantically sound manner.
""" + f"For the requirement {xml_template_formal} containing formal rules, your answer is {nat_template_formal}. For the requirement {xml_template_raw} without formal rules, your answer is {nat_template_raw}. Assertions can be expressed in a similar manner. " + """
Please strictly refer to the following knowledge snippet to answer the question: {context}
Question: {question}
Answer Requirements:
1. Only format the natural language requirements corresponding to the XML format; Markdown code blocks are not required.
2. Analyze the requirements carefully; do not omit or add any extra information.
3. The natural language format should be reasonable, distinguishing between different types of code content and fully expressing the requirements.
"""
        
xml2natAssistant = RAGAssistant(vector_db, template)

if __name__ == "__main__":
    while True:
        question = input("输入XML需求")
        res = xml2natAssistant.query(question)
        print(res["answer"])