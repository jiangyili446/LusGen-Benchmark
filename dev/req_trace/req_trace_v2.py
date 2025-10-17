import os
import time
import pandas as pd


def generate_traceability_matrix(
    # name: str,
    client,
    model: str,
    req_directory: str,
    lus_directory: str,
    output_directory: str,
):

    os.makedirs(output_directory, exist_ok=True)
    processing_records = []

    def init_message(des: str, code: str):
        return """
        ### Role:
        You are now a senior software engineer specializing in safety-critical systems. Your mission is to generate the requirement-to-design traceability matric. Any inaccuracies in traceability may lead to catastrophic failures, so rigor and precision are paramount.
        ### Knowledge:
        According to DO-178C, traceability between requirements and detailed design (including Lustre code) must adhere to:
        1. Completeness: All requirements must be traced to corresponding detailed design lines. No requirement shall be left untraced, ensuring that every specified need is addressed in the design.
        2. Bidirectional Traceability: 
            - Forward traceability: From each requirement to its corresponding Lustre design. This verifies that the design correctly implements all requirements, ensuring no requirement is omitted during design development.
            - Backward traceability: From each Lustre design line back to its originating requirements. This enables rapid impact analysis, if a lustre design line is modified, engineers can immediately identify which requirements are affected, ensuring changes do not inadvertently violate original requirements.

        ### Your Task:
        For the input requirements and Lustre, genrate traceability between requirements and Lustre line, ensuring compliance with DO-178C.
        Based on the requirements and the corresponding Lustre code, generate a traceability matrix between each requirement and the corresponding line of code.
        Answer Requirements:
        1. The answer must include the original requirements and Lustre code, as well as the generated requirement-code traceability matrix and an explanation of the traceability.
        2. The traceability matrix should be implemented using a Markdown table. Column L represents the Lustre code line number, and row R represents each requirement. A cell value of 1 indicates traceability exists, and a cell value of 0 indicates no traceability exists.
        3. The answer should only include the following sections: ### Req, ### Lustre, ### Req-Lus Trace Matrix, and ### Trace Reasons. Do not include any information that is not necessary for the task.
        4. Analyze the requirements and code thoroughly, ensuring no omissions or extra information is included.
    
        The requirements and model to generate traceability are as follows: 
        - Requirements: """ + f"{des}" + """
        - Lustre: """ + f"{code}" + """

        ### Example
        - Example Requirements: 
        定义了一个节点top，它接收一个布尔输入reset，并返回一个布尔输出OK。在节点内部，定义了一个无符号8位整数变量i。i的计算逻辑是：在每个时刻，i的值由前一个时刻的i加上1得到，初始时刻为1。OK变量用于检查i的值是否大于或等于0，并声明了一个属性，以确保这个条件始终成立。
        - Example Lustre code:
        ```
        node top (reset: bool) returns (OK: bool);
        var i : uint8;
        let
          i = (uint8 1) -> pre i + (uint8 1);
          OK = i >= (uint8 0);
          --%PROPERTY OK;
        tel
        ```
        - Example Req-Lus Trace Matrix
        | Req-Lus | L1 | L2   | L3 | L4 | L5 | L6 | L7 |
        |------|------|--------|----|----|----|----|
        | R1 | 1   | 0 | 0 | 0 | 0 | 0 | 0 |
        | R2 | 0   | 1 | 0 | 0 | 0 | 0 | 0 |
        | R3 | 0   | 0 | 0 | 1 | 0 | 0 | 0 |
        | R4 | 0   | 0 | 0 | 0 | 1 | 1 | 0 |
        """

    total_time = 0
    for root, _, files in os.walk(req_directory):
        for filename in files:
            if not filename.endswith(".txt"):
                continue
            req_path = os.path.join(root, filename)

            try:
                with open(req_path, "r", encoding="utf-8") as f:
                    des = f.read()

                base_name = filename.replace("_nat_des.txt", "")
                lus_filename = f"{base_name}.lus"
                lus_path = os.path.join(lus_directory, lus_filename)

                if not os.path.exists(lus_path):
                    print(f"❌ 找不到对应 Lustre 文件: {lus_filename}")
                    continue

                with open(lus_path, "r", encoding="utf-8") as lf:
                    code = lf.read()

                # === 调用模型生成 ===
                start = time.time()
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": init_message(des, code)}],
                )
                end = time.time()
                processing_time = round(end - start, 2)
                total_time += processing_time

                result = response.choices[0].message.content.strip()

                # === 保存输出 ===
                output_name = filename.replace("_nat_des.txt", "_trace.txt")
                output_path = os.path.join(output_directory, output_name)

                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(result + "\n")

                processing_records.append({
                    "filename": filename,
                    "lus_filename": lus_filename,
                    "processing_time(s)": processing_time,
                })
                print(f"✅ {filename} → {output_name}")

            except Exception as e:
                print(f"❌ {filename} 处理出错: {e}")

    # === 保存统计结果 ===
    csv_path = os.path.join(output_directory, f"_records.csv")
    pd.DataFrame(processing_records).to_csv(csv_path, encoding="utf-8", index=False)
    print(f"\n📊 实验  完成，总耗时 {round(total_time, 2)} 秒")
    print(f"结果已保存到: {csv_path}")


# === 示例调用 ===
if __name__ == "__main__":
    from dev.agents import gemini_client

    generate_traceability_matrix(
        client=gemini_client,
        model="gemini-2.5-flash",
        req_directory="../../e4_dataset/d2_nat_des",
        lus_directory="../../benchmark/d2/Lustre",
        output_directory="../../experimentalResults/experiment4/gemini/d2",
    )
