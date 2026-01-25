# src/models/openai_client.py
from typing import List, Dict
from openai import OpenAI

from src.settings import get_setting
from src.models.base import EmbeddingsClient, ChatClient


class OpenAIEmbeddings(EmbeddingsClient):
    def __init__(self):
        self.client = OpenAI()  # uses OPENAI_API_KEY env
        self.model = get_setting("embeddings.model_name")

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


class OpenAIChat(ChatClient):
    def __init__(self):
        self.client = OpenAI()
        self.model = get_setting("llm.model_name")

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content
