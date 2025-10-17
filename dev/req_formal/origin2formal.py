import os
from req2xml import req2xmlAssistant
from xml2nat import xml2natAssistant
import time
import pandas as pd
root_dir = os.path.dirname(os.path.dirname(__file__))
import sys
sys.path.append(root_dir)
from dev.agents import qwen_client
client = qwen_client

current_directory = os.path.dirname(os.path.abspath(__file__))

relative_directory = '../../benchmark/d5/ori_req_cn'
directory = os.path.join(current_directory, relative_directory)

current_file_path = os.path.abspath(__file__)

output_directory = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
os.makedirs(output_directory, exist_ok=True)
records = []
total_time = 0
for root, _, files in os.walk(directory):
    for filename in files:
        if filename.endswith(".txt"):
            filepath = os.path.join(root, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as file:
                    des = file.read()
                t1 = time.time()
                xml_des = req2xmlAssistant.query(des)["answer"]
                t2 = time.time()
                t_req_xml = t2 - t1
                new_filename = filename.replace(".txt", "_xml_des.txt")

                output_filepath = os.path.join(output_directory, "benchmark/d5/xml_req", new_filename)

                with open(output_filepath, 'w', encoding='utf-8') as output_file:
                    output_file.write(xml_des)

                print(f"{filename}  {output_filepath}")
                t3 = time.time()
                nat_des = xml2natAssistant.query(xml_des)["answer"]
                t4 = time.time()
                t_nat_des = t4 - t3
                new_filename = filename.replace(".txt", "_formal_des.txt")

                output_filepath = os.path.join(output_directory, "benchmark/d5/formal_req", new_filename)

                with open(output_filepath, 'w', encoding='utf-8') as output_file:
                    output_file.write(nat_des)
                print(f"{filename}  {output_filepath}")


            except Exception as e:
                print(f"{filename} ", e)

pd.DataFrame(records).to_csv(os.path.join(output_directory, "gen_req_formal_records.csv"), index=False)