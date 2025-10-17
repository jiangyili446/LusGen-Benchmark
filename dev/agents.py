from openai import OpenAI
from dev.const import qwen_apiKey, gpt_apiKey, deepseek_apiKey, gemini_apiKey
from google import genai

qwen_client = OpenAI(
    api_key=qwen_apiKey,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

gpt_client = OpenAI(
    api_key=gpt_apiKey
)

deepseek_client = OpenAI(
    api_key=deepseek_apiKey,
    base_url="https://api.deepseek.com"
)



gemini_client = OpenAI(
    api_key=gemini_apiKey,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
