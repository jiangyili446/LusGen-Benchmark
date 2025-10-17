import os
root_dir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(root_dir)
from dev.RAG.RAGAssistant import RAGAssistant
from dev.RAG.KnowledgeBuilder import KnowledgeBuilder
from langchain_community.vectorstores import FAISS
from dev.const import gpt_apiKey, qwen_apiKey
os.environ["OPENAI_API_KEY"] = gpt_apiKey
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey

req_template_formal = """
Define two nodes, Node X and Node Y.
Node X receives a boolean input i and returns an integer x. Node X internally defines a variable OK, which is used to indicate the validity of the state. The calculation logic for x is as follows: it is 0 at the initial moment. If the value of x at the previous moment is less than 2, then the value of x is 1 minus the value of x at the previous moment; if the previous condition is not met and the input i is true, then the value of x is 3; in other cases, the value of x is set to 2. In addition, the variable OK is used to check whether the value of x is less than 3, and a kind2 verification entry is declared to ensure that this condition is always met.
Node Y receives two boolean inputs i and clk, and returns an integer x. Node Y defines a variable OK. The calculation logic for x is: when clk is true, Node Y calls Node X and uses i as the input; when clk is false, the value of x is 0. The OK variable of Node Y is also used to check whether x is less than 3, and a kind2 verification entry is declared to ensure that this condition holds in all cases.
"""
xml_template_formal = """
<Requirement>
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
        <Func>The initial x is 0. If the previous x is less than 2, then the value of x is set to 1 minus the previous x. If the previous condition is not met and input i is true, then the value of x is 3. In other cases, the value of x is set to 2. </Func>
        <Validations>
            <Rule expression="OK">x is less than 3. </Rule>
        </Validations>
    </Node>
    <Node name="Y">
        <Inputs>
            <Param name="i" type="bool"/>
            <Param name="clk" type="bool"/>
        </Inputs>
        <Outputs>
            <Param name="x" type="int"/>
        </Outputs>
        <Vars>
            <Var name="OK" type="bool"/>
        </Vars>
            <Func>When clk is true, node Y calls node X with i as input; when clk is false, the value of x is set to 0. </Func>
        <Validations>
            <Rule expression="OK">x is less than 3. </Rule>
        </Validations>
    </Node>
</Requirement>
"""

req_template_raw = """
This node defines a First node that accepts an integer input, X, and returns an integer, Y. The calculation logic for Y is as follows: Y's value always remains at the current input, X, while the next time step uses the value of Y at the previous time step. This creates a constant flow, where Y's value always remains at the initial value of X.
"""

xml_template_raw = """
    <Requirement>
        <Node name="First">
        <Inputs>
            <Param name="X" type="int"/>
        </Inputs>
        <Outputs>
            <Param name="Y" type="int"/>
        </Outputs>
        <Func>The value of Y always remains the current input value of X, while the value of Y at the next moment is used. </Func>
    </Node>
</Requirement>
"""

xml_template_func = """
<Function name="lookup_table">
    <Inputs>
        <Param name="i" type="int"/>
    </Inputs>
    <Outputs>
        <Param name="v" type="int"/>
    </Outputs>
    <Func>v is the value of v at the previous moment plus i</Func>
</Function>
"""

xml_template_const = """
<Const name="array" type="int^3" value="[1, 5, 9]"/>
"""

req_template_type = """
Defines a type foo. It also defines a type alias representing an integer type. It also defines a structure type pair with two members: a, of type alias; and b, of type integer. It also defines an enumeration type color with three possible values: blue, white, and black.
"""

xml_template_type = """
<Type name="foo"/>
<Type name="alias" base="int"/>
<Type name="pair">
    <Struct>
    <Field name="a" type="alias"/>
    <Field name="b" type="int"/>
    </Struct>
</Type>
<Type name="color">
    <Enum>
    <Value name="blue"/>
    <Value name="white"/>
    <Value name="black"/>
    </Enum>
</Type>
"""

template = """
You are a Lustre language expert and familiar with its kind2 formal verification tool. Your task is to analyze user-entered requirements and output them in XML format.
""" + f"For the requirement {req_template_formal} containing formal rules, your answer is {xml_template_formal}. For the requirement {req_template_raw} without formal rules, your answer is {xml_template_raw}. For functions, the XML format reference is {xml_template_func}; for constants, the XML format reference is {xml_template_const}; for type definition requirements {req_template_type}, the XML format reference is {xml_template_type}" + """
Please strictly refer to the following knowledge segment to answer the question: {context}
Question: {question}
Answer requirements:
1. Your answer must only include the XML format corresponding to the requirement; markdown code blocks are not required.
2. Analyze the requirement content carefully; do not omit or add any additional information. 3. The XML format should be reasonable, differentiating different types of code content and fully expressing the required content.
"""

builder = KnowledgeBuilder()
current_file = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file)
save_path = "../RAG/vector_store"
target_dir = os.path.join(parent_dir, save_path)
# target_dir = "D:\\project\\LLM4Lustre\\dev\\RAG\\vector_store"
vector_db = FAISS.load_local(target_dir, embeddings=builder.embeddings,
allow_dangerous_deserialization=True) 
req2xmlAssistant = RAGAssistant(vector_db, template)

if __name__ == "__main__":
    while True:
        question = input("input req:")
        res = req2xmlAssistant.query(question)
        print(res["answer"])