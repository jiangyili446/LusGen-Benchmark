import os
import time
import subprocess
import pandas as pd
from dev.gen.prompt.get_prompt import init_prompt, syntax_feedback_prompt


def run_lv6_experiment(
    name: str,
    client,
    directory: str,
    output_directory: str,
    tmp_file_linux: str,
    model: str,
    max_retries: int = 5,
):
    """
    参数化实验函数：
    根据自然语言描述生成 Lustre 代码，使用 lv6 工具进行语法检查，
    若检查失败则根据错误信息让模型进行修复，直到成功或达到最大重试次数。

    Args:
        name (str): 实验名称（如 "gemini_t1"）
        client: LLM 客户端实例（如 gemini_client, qwen_client, gpt_client）
        directory (str): 输入自然语言文件目录
        output_directory (str): 输出目录
        tmp_file_linux (str): lv6 检查用的 Lustre 文件路径（Linux 路径）
        model (str): 使用的模型名
        max_retries (int): 最大修复次数
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

                # === 第一次生成 Lustre 代码 ===
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": init_prompt(des)},
                    ],
                )
                code = response.choices[0].message.content.strip()
                record = {
                    "experiment": name,
                    "filename": filename,
                    "description": des,
                    "code_0": code,
                }

                # === 语法检查与反馈修复循环 ===
                for i in range(max_retries):
                    tmp_lus_local = os.path.join(current_directory, "tmp.lus")
                    with open(tmp_lus_local, "w", encoding="utf-8") as tmp_file:
                        tmp_file.write(code)

                    try:
                        result = subprocess.run(
                            [
                                "wsl.exe",
                                "/home/xxxx/.opam/default/bin/lv6",
                                tmp_file_linux,
                            ],
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                        )
                        record["returncode"] = result.returncode

                        if result.returncode == 0:
                            print(f"✅ {filename} 第{i+1}次 LV6 语法检查通过")
                            break
                        else:
                            feedback = result.stderr.decode("utf-8", errors="ignore")
                            print(f"⚠️ {filename} 第{i+1}次语法错误，模型生成修复中…")

                            response = client.chat.completions.create(
                                model=model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": syntax_feedback_prompt(des, code, feedback),
                                    }
                                ],
                            )
                            code = response.choices[0].message.content.strip()
                            record[f"code_{i+1}"] = code

                    except Exception as e:
                        print(f"❌ LV6 执行异常: {e}")
                        break

                end = time.time()
                gen_time = round(end - start, 2)
                total_time += gen_time
                record["gen_time"] = gen_time
                records.append(record)

                # === 保存最终代码 ===
                new_filename = filename.replace("_des.txt", "_gen_lv6.lus")
                output_filepath = os.path.join(output_directory, new_filename)
                with open(output_filepath, "w", encoding="utf-8") as out:
                    out.write(code)

                print(f"💾 {filename} 最终结果已保存到 {output_filepath}")

            except Exception as e:
                print(f"❌ {filename} 处理出错: {e}")

    # === 保存实验结果 ===
    result_csv = os.path.join(output_directory, f"{name}_records.csv")
    pd.DataFrame(records).to_csv(result_csv, encoding="utf-8", index=False)
    print(f"\n✅ 实验 {name} 完成，共 {len(records)} 个文件，总耗时 {round(total_time, 2)} 秒")
    print(f"📊 记录已保存到: {result_csv}")


# === 示例调用 ===
if __name__ == "__main__":
    from dev.agents import gemini_client, deepseek_client

    run_lv6_experiment(
        name="gemini_lv6_d4",
        client=deepseek_client,
        directory="../../formal_req/d4_nat_des",
        output_directory="../../test",
        tmp_file_linux="/mnt/c/Users/18704/Desktop/2026-AAAI-LusGen/LusGen/dev/gen/tmp.lus",
        model="deepseek-chat",
        max_retries=5,
    )
