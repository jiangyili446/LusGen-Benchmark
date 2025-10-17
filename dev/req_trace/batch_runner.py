import yaml
from openai import OpenAI

from dev.agents import deepseek_client, gpt_client, qwen_client
from dev.req_trace.req_trace_v2 import generate_traceability_matrix

gemini_client = OpenAI(
    api_key="AIzaSyD2F-MlWaBpNCxMZCrj1-9lBRyAUS8R7hE",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


CLIENT_MAP = {
    "gemini": gemini_client,
    "deepseek": deepseek_client,
    "gpt": gpt_client,
    "qwen": qwen_client
}

if __name__ == "__main__":
    with open("replicate_trace.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for exp in cfg["experiments"]:
        client_name = exp.pop("client")
        exp_name = exp.pop("name", None)  # <== 这行加上
        client = CLIENT_MAP[client_name]
        print(f"\n🚀 Running {exp_name or client_name} with client: {client_name}")
        generate_traceability_matrix(client=client, **exp)

