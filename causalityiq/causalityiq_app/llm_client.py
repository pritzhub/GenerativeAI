# causalityiq_app/llm_client.py
from __future__ import annotations

import os

from openai import OpenAI  # pip install openai [web:156]
from causalityiq_app.settings import settings

import json
import logging

logger = logging.getLogger(__name__)

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
  global _client
  if _client is None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
      raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    _client = OpenAI(api_key=api_key)  # [web:156]
  return _client


def generate_incident_analysis(
  incident_text: str,
  analysis_instructions: str | None,
  tone: str | None,
  constraints: str | None,
) -> str:
  """
  Call the chat completion model configured under llm.models.analysis.
  Returns the model's analysis as plain text.
  """
  client = get_openai_client()
  model_name = settings.llm.models.analysis

  # Build a concise system + user prompt from pieces. [web:153][web:159]
  system_parts: list[str] = [
    "You are an expert incident analyst.",
    "You receive incident context and must produce a clear, structured analysis.",
  ]
  if tone:
    system_parts.append(f"Use this tone/style: {tone}.")
  if constraints:
    system_parts.append(f"Follow these constraints: {constraints}.")

  system_prompt = " ".join(system_parts)

  user_prompt_parts: list[str] = []
  if analysis_instructions:
    user_prompt_parts.append(f"Instructions: {analysis_instructions}")
  user_prompt_parts.append("Incident context:")
  user_prompt_parts.append(incident_text)

  user_prompt = "\n\n".join(user_prompt_parts)

  messages = [system_prompt, user_prompt]

  if settings.debug_log_llm_prompts:
    logger.info("LLM incident analysis messages:\n%s", json.dumps(messages, indent=2, ensure_ascii=False),)
        
  resp = client.chat.completions.create(  # 
    model=model_name,
    messages=[
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": user_prompt},
    ],
    temperature=settings.llm.params.temperature,
    max_tokens=settings.llm.params.max_tokens,
  )

  content = resp.choices[0].message.content or ""
  return content.strip()
