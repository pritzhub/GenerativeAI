# src/models/base.py
from typing import List, Dict, Any


class EmbeddingsClient:
    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


class ChatClient:
    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
        raise NotImplementedError
