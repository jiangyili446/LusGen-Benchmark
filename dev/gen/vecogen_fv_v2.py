import os
import time
import re
import subprocess
import pandas as pd
from prompt.get_prompt import verify_feedback_prompt, init_prompt


def clean_ansi_codes(text: str) -> str:
    """去除命令行ANSI控制字符"""
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def run_experiment_veco(
    client,
    directory: str,
    output_directory: str,
    tmp_file_linux: str,
    model: str,
    max_retries: int = 5,
    timeout: int = 30,
):
    """
    参数化实验函数：
    根据自然语言描述生成 Lustre 代码，调用 Kind2 验证器验证，
    若验证失败则根据反馈进行迭代修正，直到通过或达最大次数。

    Args:
        name (str): 实验名称（如 "gemini_t1"）
        client: LLM 客户端实例（如 gemini_client, qwen_client, gpt_client）
        directory (str): 输入自然语言文件目录
        output_directory (str): 输出目录
        tmp_file_linux (str): WSL 中 Kind2 使用的 Lustre 文件路径
        model (str): 使用的模型名
        max_retries (int): 最大验证修复次数
        timeout (int): Kind2 超时时间（秒）
    """

    os.makedirs(output_directory, exist_ok=True)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    total_time = 0
    records = []

    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue

            filepath = os.path.join(root, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    des = f.read()

                start = time.time()
                print(f"\n🚀 正在处理文件: {filename}")

                # === 第一次生成代码 ===
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": init_prompt(des)},
                    ],
                )
                code = response.choices[0].message.content.strip()
                record = {
                    "filename": filename,
                    "description": des,
                    "code_0": code,
                }

                # === 验证与反馈循环 ===
                for i in range(max_retries):
                    tmp_lus_local = os.path.join(current_directory, "tmp_fv.lus")
                    with open(tmp_lus_local, "w", encoding="utf-8") as tmp_file:
                        tmp_file.write(code)

                    try:
                        result = subprocess.run(
                            [
                                "wsl.exe",
                                "/home/xxxx/.opam/default/bin/kind2",
                                tmp_file_linux,
                                "--z3_bin",
                                "/home/xxxx/.opam/default/bin/z3",
                            ],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=timeout,
                        )

                        feedback = clean_ansi_codes(result.stdout.decode("utf-8", errors="ignore"))
                        record["returncode"] = result.returncode

                        if result.returncode in (0, 20):
                            print(f"✅ {filename} 第{i+1}次验证通过")
                            break
                        else:
                            print(f"⚠️ {filename} 第{i+1}次验证失败，生成反馈修正")
                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": verify_feedback_prompt(des, code, feedback),
                                    }
                                ],
                            )
                            code = response.choices[0].message.content.strip()
                            record[f"code_{i+1}"] = code

                    except subprocess.TimeoutExpired:
                        print(f"⏰ Kind2 验证超时 ({timeout}s)")
                        break
                    except Exception as e:
                        print(f"❌ 验证阶段异常: {e}")
                        break

                end = time.time()
                gen_time = round(end - start, 2)
                total_time += gen_time
                record["gen_time"] = gen_time
                records.append(record)

                # === 保存最终结果 ===
                new_filename = filename.replace("_des.txt", "_gen_kind2.lus")
                output_filepath = os.path.join(output_directory, new_filename)
                with open(output_filepath, "w", encoding="utf-8") as out:
                    out.write(code)

                print(f"💾 {filename} 最终结果已保存到 {output_filepath}")

            except Exception as e:
                print(f"❌ {filename} 处理出错: {e}")

    # === 保存所有实验记录 ===
    result_csv = os.path.join(output_directory, f"_records.csv")
    pd.DataFrame(records).to_csv(result_csv, encoding="utf-8", index=False)
    print(f"\n✅ 实验  完成，共 {len(records)} 个文件，总耗时 {round(total_time, 2)} 秒")
    print(f"📊 记录已保存到: {result_csv}")


# === 示例调用 ===
if __name__ == "__main__":
    from dev.agents import gemini_client, deepseek_client

    run_experiment_veco(
        name="gemini_t1",
        client=deepseek_client,
        directory="../../benchmark/d4/ori_req_prop",
        output_directory="../../test",
        tmp_file_linux="/mnt/c/Users/18704/Desktop/2026-AAAI-LusGen/LusGen/dev/gen/tmp_fv.lus",
        model="deepseek-chat",
        max_retries=5,
        timeout=30,
    )
