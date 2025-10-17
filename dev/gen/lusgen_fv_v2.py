import os
import time
import subprocess
import pandas as pd
import re
from dev.gen.prompt.get_prompt import rag_prompt, rag_verify_feedback_prompt
from dev.process_lustre import process_lus_file
from dev.RAG.KnowledgeBuilder import KnowledgeBuilder
from dev.const import gpt_apiKey, qwen_apiKey
from langchain_community.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = gpt_apiKey
os.environ["DASHSCOPE_API_KEY"] = qwen_apiKey


def clean_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def retrieve_context(question: str, retriever, k: int = 5):
    docs = retriever.get_relevant_documents(question)
    return [doc.page_content for doc in docs]


def run_experiment_lusgen(
    client,
    directory,
    output_directory,
    save_path,
    tmp_file_linux,
    model,
    max_retries,
    timeout=30
):
    """主执行函数，可配置各类参数"""

    current_directory = os.path.dirname(os.path.abspath(__file__))

    # 路径转为绝对路径
    directory = os.path.abspath(directory)
    output_directory = os.path.abspath(output_directory)
    save_path = os.path.abspath(save_path)
    tmp_file_linux = tmp_file_linux

    os.makedirs(output_directory, exist_ok=True)

    # 构建知识检索器
    builder = KnowledgeBuilder()
    vector_db = FAISS.load_local(save_path, embeddings=builder.embeddings,
                                 allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"k": 5})

    total_time = 0
    records = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)
            print("-------------------------------------")
            print(filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    des = file.read()

                start = time.time()
                context = retrieve_context(des, retriever, k=1)
                # print(context)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": rag_prompt(des, context)}
                    ]
                )

                code = response.choices[0].message.content.strip()
                record = {"filename": filename, "des": des, "code_0": code}

                for i in range(max_retries):
                    print(f"Time{i}")
                    tmp_lus = os.path.join(current_directory, 'tmp_fv.lus')
                    with open(tmp_lus, 'w', encoding='utf-8') as tmp_file:
                        tmp_file.write(code)

                    process_lus_file(tmp_lus)
                    try:
                        result = subprocess.run(
                            ["wsl.exe", "/home/xxxx/.opam/default/bin/kind2", tmp_file_linux, "--z3_bin", "/home/xxxx/.opam/default/bin/z3"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=timeout
                        )
                        record["returncode"] = result.returncode
                        feedback = clean_ansi_codes(result.stdout.decode("utf-8", errors="ignore"))
                        # print(feedback)
                        # success_count = feedback.count("<Success>")
                        # failure_count = feedback.count("<Failure>")
                        # print(f"{success_count}条通过；{failure_count}条未通过")
                        if result.returncode in (0, 20):
                            break
                        else:
                            context = retrieve_context(des + feedback, retriever, k=1)
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {"role": "user", "content": rag_verify_feedback_prompt(des, code, feedback, context)}
                                ]
                            )
                            code = response.choices[0].message.content.strip()
                            record[f"code_{i+1}"] = code

                    except subprocess.TimeoutExpired:
                        break
                    except Exception as e:
                        print("Error during subprocess:", e)

                end = time.time()
                gen_time = end - start
                total_time += gen_time
                record["gen_time"] = round(gen_time, 2)
                records.append(record)

                new_filename = filename.replace("_des.txt", "_gen_kind2.lus")
                output_filepath = os.path.join(output_directory, new_filename)
                with open(output_filepath, 'w', encoding='utf-8') as output_file:
                    output_file.write(code)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    pd.DataFrame(records).to_csv(os.path.join(output_directory, 'gen_kind2_records.csv'), encoding="utf-8")


if __name__ == "__main__":
    from dev.agents import gemini_client, deepseek_client  # 可切换到 deepseek_client / gpt_client
    # run_experiment_lusgen(
    #     client=deepseek_client,
    #     directory='../../formal_req/d2_nat_des',
    #     output_directory='../../ExperimentalResults/test',
    #     save_path='../RAG/vector_store',
    #     tmp_file_linux='/mnt/d/Project/LusGen-FASE/dev/gen/tmp_fv.lus',
    #     model='deepseek-chat',
    #     max_retries=5
    # )
    run_experiment_lusgen(
        client=deepseek_client,
        directory='../../benchmark/d5/formal_req',
        output_directory='../../ExperimentalResults/test/d5',
        save_path='../RAG/vector_store',
        tmp_file_linux='/mnt/d/Project/LusGen-FASE/dev/gen/tmp_fv.lus',
        model='deepseek-chat',
        max_retries=5
    )
