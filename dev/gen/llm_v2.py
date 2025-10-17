import os
import time
import pandas as pd
from dev.gen.prompt.get_prompt import rag_prompt, rag_syntax_feedback_prompt, init_prompt
from dev.RAG.KnowledgeBuilder import KnowledgeBuilder
from dev.const import gpt_apiKey, qwen_apiKey
from langchain_community.vectorstores import FAISS

# 设置 API Key
os.environ["OPENAI_API_KEY"] = gpt_apiKey
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey


def build_retriever():
    """加载向量数据库"""
    builder = KnowledgeBuilder()
    current_file = os.path.abspath(__file__)
    parent_dir = os.path.dirname(current_file)
    save_path = "../RAG/vector_store"
    target_dir = os.path.join(parent_dir, save_path)
    vector_db = FAISS.load_local(
        target_dir, embeddings=builder.embeddings, allow_dangerous_deserialization=True
    )
    return vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})


def retrieve_context(question: str, retriever, k: int = 5):
    docs = retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]


def run_experiment_llm(
        # name: str,
        client,
        directory: str,
        output_directory: str,
        model: str
):
    """
    参数化实验函数，用于根据自然语言描述生成 Lustre 代码。

    Args:
        name (str): 实验名称（如 'qwen_t1'）
        client: LLM 客户端实例（如 gemini_client, qwen_client, gpt_client）
        directory (str): 输入文件目录（存放自然语言描述）
        output_directory (str): 输出文件目录
        model (str): 使用的模型名
    """
    os.makedirs(output_directory, exist_ok=True)
    retriever = build_retriever()

    total_time = 0
    records = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)

            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    des = file.read()

                start = time.time()

                response = client.chat.completions.create(
                    model = model,
                    messages=[
                        {"role": "system",
                         "content": f"你是一个精通Lustre语言的专家，熟知Lustre代码的各种特性。同时了解Kind2形式化验证工具的用法。请根据自然语言描述，生成对应的lustre代码，及注释形式的Kind2性质验证入口。注意你的回答只需要给出代码，不需要给出任何多余信息或者markdown代码块的格式"},
                        {
                            "role": "user",
                            "content": f"根据自然语言描述，生成Lustre代码,\n{des}"
                        }
                    ]
                )

                code = response.choices[0].message.content.strip()
                record = {
                    "filename": filename,
                    "des": des,
                    "code_0": code
                }

                end = time.time()
                gen_time = end - start
                total_time += gen_time
                record["gen_time"] = round(gen_time, 2)
                records.append(record)

                new_filename = filename.replace("_des.txt", "_gen.lus")

                output_filepath = os.path.join(output_directory, new_filename)

                with open(output_filepath, 'w', encoding='utf-8') as output_file:
                    output_file.write(code)

                print(f"{filename} saved to {output_filepath}")

            except Exception as e:
                print(f"{filename} ", e)


    # 导出实验记录
    result_csv = os.path.join(output_directory, f"_records.csv")
    pd.DataFrame(records).to_csv(result_csv, encoding="utf-8", index=False)
    print(f"\n✅ 实验  完成，共 {len(records)} 个文件，总耗时 {round(total_time, 2)} 秒")
    print(f"结果已保存至: {result_csv}")

# ✅ 示例调用：
if __name__ == "__main__":
    from dev.agents import gemini_client, deepseek_client, qwen_client

    run_experiment_llm(
        name="gemini_t1",
        client=deepseek_client,
        directory="../../formal_req/d2_nat_des",
        output_directory="../../ExperimentalResults/Replicate/d2/qwen/d2/1",
        # model="qwen2.5-coder-32b-instruct",
        model="deepseek-chat"
    )
