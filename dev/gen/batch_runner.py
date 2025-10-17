import yaml
from openai import OpenAI

from dev.agents import deepseek_client, gpt_client, qwen_client, gemini_client
from dev.gen.lusgen_fv_v2 import run_experiment_lusgen
from dev.gen.llm_v2 import run_experiment_llm
from dev.gen.vecogen_fv_v2 import run_experiment_veco


CLIENT_MAP = {
    "gemini": gemini_client,
    "deepseek": deepseek_client,
    "gpt": gpt_client,
    "qwen": qwen_client
}

if __name__ == "__main__":
    with open("replicate_lusgen.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for exp in cfg["experiments"]:
        client_name = exp.pop("client")
        exp_name = exp.pop("name", None)  # <== 这行加上
        client = CLIENT_MAP[client_name]
        print(f"\n🚀 Running {exp_name or client_name} with client: {client_name}")
        run_experiment_lusgen(client=client, **exp)

