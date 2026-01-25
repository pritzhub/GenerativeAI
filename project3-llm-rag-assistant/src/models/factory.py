# src/models/factory.py
from src.settings import get_setting
from src.models.base import EmbeddingsClient, ChatClient
from src.models.openai_client import OpenAIEmbeddings, OpenAIChat
# later: import OllamaEmbeddings, OllamaChat, AnthropicChat, etc.


def get_embeddings_client() -> EmbeddingsClient:
    provider = get_setting("embeddings.provider")
    if provider == "openai":
        return OpenAIEmbeddings()
    # elif provider == "ollama": return OllamaEmbeddings()
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def get_chat_client() -> ChatClient:
    provider = get_setting("llm.provider")
    if provider == "openai":
        return OpenAIChat()
    # elif provider == "ollama": return OllamaChat()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
