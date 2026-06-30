import re
import os
import subprocess
import time
import pandas as pd
from typing import List, Dict, Any

# --- 常量定义 ---
# 统一的平均数计算基数
STANDARD_DIVISOR = 77

# --- 环境和路径设置 ---
# 待处理的 Windows 根目录
win_root_dir = 'D:\\Project\\LusGen-Benchmark\\benchmark'
# 对应的 Linux/WSL 根目录
linux_root_dir = '/mnt/d/Project/LusGen-Benchmark/benchmark'

# 用于存储每个文件的详细执行记录
detailed_records: List[Dict[str, Any]] = []
# 用于存储每个目录的统计摘要
directory_summaries: List[Dict[str, Any]] = []


def clean_ansi_codes(text: str) -> str:
    """移除 ANSI 控制字符"""
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def process_directory_recursive(win_dir: str, linux_dir: str):
    """递归遍历目录，并按子目录统计 <Success> 数量"""

    overall_success_count = 0
    overall_file_count = 0

    # os.walk 会递归遍历所有子目录
    for root, _, files in os.walk(win_dir):

        # 1. 初始化当前目录的统计数据
        current_success_count = 0
        current_failure_count = 0
        current_file_count = 0

        # 计算当前目录相对于 Windows 根目录的相对路径
        relative_path = os.path.relpath(root, win_dir)

        # 将 Windows 的相对路径转换为 Linux 格式（用 / 替换 os.sep）
        linux_relative_path = relative_path.replace(os.sep, '/')

        # 构造当前目录在 Linux 环境下的绝对路径
        linux_current_dir = os.path.join(linux_dir, linux_relative_path).replace(os.sep, '/')

        lus_files = [f for f in files if f.endswith(".lus")]

        if not lus_files:
            continue

        print(f"\n--- 正在处理目录: {relative_path} ---")

        # 2. 循环处理当前目录下的所有 .lus 文件
        for filename in lus_files:

            # 构造文件在 Linux 环境下的完整路径
            linux_filepath = os.path.join(linux_current_dir, filename).replace(os.sep, '/')

            current_file_count += 1

            # --- 执行 kind2 检查 ---
            try:
                result = subprocess.run(
                    ["wsl.exe", "/home/yili/.opam/default/bin/kind2", linux_filepath, "--z3_bin",
                     "/home/yili/.opam/default/bin/z3"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=120
                )

                rc = result.returncode
                stdout_str = result.stdout.decode("utf-8", errors="ignore")
                stdout_str = clean_ansi_codes(stdout_str)

                # print(stdout_str)

                # 统计成功次数
                success_count_file = stdout_str.count("<Success>")
                failure_count_file = stdout_str.count("<Failure>")
                current_success_count += success_count_file
                current_failure_count += failure_count_file

                returncode_simple = 0 if rc in [0, 20] else 1

                # 记录详细信息
                detailed_records.append({
                    "directory": relative_path if relative_path != '.' else '(Top_Level)',
                    "filename": filename,
                    "returncode_kind2": rc,
                    "returncode_simple": returncode_simple,
                    "success_count": success_count_file,
                    "failure_count": failure_count_file,
                })

                print(f"{filename}: success={success_count_file}, failure={failure_count_file}, rc={rc}")

            except subprocess.TimeoutExpired:
                print(f"Warning: {filename} timed out.")
                detailed_records.append({
                    "directory": relative_path if relative_path != '.' else '(Top_Level)',
                    "filename": filename,
                    "returncode_kind2": 'TIMEOUT',
                    "returncode_simple": 1,
                    "success_count": 0,
                    "failure_count": 0,
                })
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                detailed_records.append({
                    "directory": relative_path if relative_path != '.' else '(Top_Level)',
                    "filename": filename,
                    "returncode_kind2": 'ERROR',
                    "returncode_simple": 1,
                    "success_count": 0,
                    "failure_count": 0,
                })

        # 3. 存储当前目录的统计摘要
        if current_file_count > 0:
            # --- 关键修改：标准化平均数计算 ---
            standard_avg_success = current_success_count / STANDARD_DIVISOR
            # 实际平均数（可选，用于调试）
            actual_avg_success = current_success_count / current_file_count

            directory_summaries.append({
                "Directory": relative_path if relative_path != '.' else '(Top_Level)',
                "Files_Processed": current_file_count,
                "Success_Count": current_success_count,
                "Failure_Count": current_failure_count,
                "Standard_Avg_Success": standard_avg_success,  # 新增标准化指标
                "Actual_Avg_Success": actual_avg_success  # 保留实际指标
            })

            # 更新总体统计
            overall_success_count += current_success_count
            overall_file_count += current_file_count

            # 4. 打印当前目录的统计结果
            print(f"\n--- 目录统计 ({relative_path}) ---")
            print(f"处理文件数: {current_file_count}")
            print(f"总成功数: {current_success_count}")
            print(f"总Fail数: {current_failure_count}")
            print(f"实际平均成功数: {actual_avg_success:.4f}")
            print(f"标准化平均成功数 (除以 {STANDARD_DIVISOR}): {standard_avg_success:.4f}")

    # 5. 生成和保存统计报告

    # pd.DataFrame(detailed_records).to_csv(
    #     os.path.join(win_dir, 'gen_rag_fv_kind_detailed_records.csv'),
    #     encoding="utf-8",
    #     index=False
    # )

    df_summary = pd.DataFrame(directory_summaries)
    df_summary.to_csv(
        os.path.join(win_dir, 'gpt_lusgen_prop_summary.csv'),
        encoding="utf-8",
        index=False
    )

    # 计算并打印总体统计
    # 总体标准化平均数 (按您的要求，也除以 77)
    overall_standard_avg = overall_success_count / STANDARD_DIVISOR
    # 总体实际平均数 (总成功数 / 总文件数)
    overall_actual_avg = overall_success_count / overall_file_count if overall_file_count > 0 else 0.0

    print("\n===================================")
    print("--- 总体统计摘要 ---")
    print(f"总处理文件数: {overall_file_count}")
    print(f"总成功数: {overall_success_count}")
    print(f"总体实际平均成功数: {overall_actual_avg:.4f}")
    print(f"总体标准化平均成功数 (除以 {STANDARD_DIVISOR}): {overall_standard_avg:.4f}")
    print("===================================")


if __name__ == "__main__":
    process_directory_recursive(win_root_dir, linux_root_dir)