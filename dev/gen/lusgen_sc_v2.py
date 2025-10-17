import os
import time
import subprocess
import pandas as pd
from dev.RAG.KnowledgeBuilder import KnowledgeBuilder
from langchain_community.vectorstores import FAISS

from dev.const import qwen_apiKey, gpt_apiKey
from dev.gen.prompt.get_prompt import rag_prompt, rag_syntax_feedback_prompt

os.environ["OPENAI_API_KEY"] = gpt_apiKey
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey




def run_rag_lv6_experiment(
    name: str,
    client,
    directory: str,
    output_directory: str,
    save_path: str,
    tmp_file_linux: str,
    model: str,
    max_retries: int = 5,
    k: int = 5,
):
    """
    参数化实验函数：
    使用 RAG 检索上下文信息 + LLM 生成 Lustre 代码，
    然后通过 LV6 检查代码语法，若失败则结合反馈进行修复。

    Args:
        name (str): 实验名称
        client: LLM 客户端实例
        directory (str): 输入自然语言文件目录
        output_directory (str): 输出目录
        save_path (str): 向量数据库保存路径
        tmp_file_linux (str): LV6 检查用的 Lustre 文件路径（Linux 路径）
        model (str): 使用的模型名
        max_retries (int): 最大重试次数
        k (int): RAG 检索相关文档数量
    """
    os.makedirs(output_directory, exist_ok=True)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # === 加载向量数据库 ===
    target_dir = os.path.join(current_directory, save_path)
    builder = KnowledgeBuilder()
    vector_db = FAISS.load_local(
        target_dir,
        embeddings=builder.embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": k})

    def retrieve_context(question: str):
        docs = retriever.get_relevant_documents(question)
        return [doc.page_content for doc in docs]

    total_time = 0
    records = []

    # === 遍历输入目录 ===
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    des = f.read()

                start = time.time()
                context = retrieve_context(des)

                # === 第一次生成代码 ===
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": rag_prompt(des, context)}
                    ],
                )
                code = response.choices[0].message.content.strip()
                record = {"experiment": name, "filename": filename, "description": des, "code_0": code}

                # === LV6 检查 + 修复循环 ===
                for i in range(max_retries):
                    tmp_lus_local = os.path.join(current_directory, "tmp.lus")
                    with open(tmp_lus_local, "w", encoding="utf-8") as tmp_file:
                        tmp_file.write(code)

                    try:
                        result = subprocess.run(
                            ["wsl.exe", "/home/xxxx/.opam/default/bin/lv6", tmp_file_linux],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            encoding="utf-8",
                            errors="ignore",
                        )
                        record["returncode"] = result.returncode

                        if result.returncode == 0:
                            break
                        else:
                            feedback = result.stderr
                            print(f"⚠️ {filename} 第{i+1}次 LV6 检查失败")
                            context = retrieve_context(des + feedback)
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "user", "content": rag_syntax_feedback_prompt(des, code, feedback, context)}
                                ]
                            )
                            code = response.choices[0].message.content.strip()
                            record[f"code_{i+1}"] = code

                    except Exception as e:
                        print(f"❌ LV6 执行异常: {e}")
                        break

                end = time.time()
                record["gen_time"] = round(end - start, 2)
                total_time += end - start
                records.append(record)

                # === 保存最终 Lustre 代码 ===
                new_filename = filename.replace("_des.txt", "_gen_lv6.lus")
                output_filepath = os.path.join(output_directory, new_filename)
                with open(output_filepath, "w", encoding="utf-8") as out:
                    out.write(code)

                print(f"💾 {filename} 保存到 {output_filepath}")

            except Exception as e:
                print(f"❌ {filename} 处理出错: {e}")

    # === 保存实验记录 CSV ===
    result_csv = os.path.join(output_directory, f"{name}_records.csv")
    pd.DataFrame(records).to_csv(result_csv, encoding="utf-8", index=False)
    print(f"\n✅ 实验 {name} 完成，总耗时 {round(total_time,2)} 秒")
    print(f"📊 记录保存到: {result_csv}")


# === 示例调用 ===
if __name__ == "__main__":
    from dev.agents import gemini_client, deepseek_client

    os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey
    run_rag_lv6_experiment(
        name="gemini_rag_lv6_d4",
        client=deepseek_client,
        directory="../../formal_req/d3_nat_des",
        output_directory="../../test",
        save_path="../RAG/vector_store",
        tmp_file_linux="/mnt/c/Users/18704/Desktop/2026-AAAI-LusGen/LusGen/dev/gen/tmp.lus",
        model="deepseek-chat",
        max_retries=5,
        k=5,
    )
