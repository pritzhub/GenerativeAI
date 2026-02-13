# src/settings.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class LLMParams(BaseModel):
  temperature: float = 0.2
  max_tokens: int = 2000
  timeout_seconds: int = 60


class LLMModels(BaseModel):
  analysis: str
  summarization: str
  embedding: str


class LLMConfig(BaseModel):
  provider: str
  models: LLMModels
  params: LLMParams


class StorageConfig(BaseModel):
  base_data_dir: Path = Field(default=Path("./data"))
  users_subdir: str = "users"

  @property
  def users_base_dir(self) -> Path:
    return self.base_data_dir / self.users_subdir


class LoggingRotationConfig(BaseModel):
  enabled: bool = True
  max_bytes: int = 1_048_576
  backup_count: int = 15
  encoding: str = "utf-8"


class LoggingConfig(BaseModel):
  level: str = "INFO"
  file: Path = Field(default=Path("./logs/causalityiq.log"))
  rotation: LoggingRotationConfig = LoggingRotationConfig()


class AppConfig(BaseModel):
  name: str = "CausalityIQ"
  environment: str = "dev"


class TemplatesConfig(BaseModel):
  default_report: Dict[str, Any] = Field(default_factory=dict)


class WorkflowsConfig(BaseModel):
  default_analysis_flow: str = "standard_analysis_v1"


class Settings(BaseModel):
  app: AppConfig
  llm: LLMConfig
  storage: StorageConfig
  workflows: WorkflowsConfig
  templates: TemplatesConfig
  logging: LoggingConfig
  debug_log_llm_prompts: bool = False


def _load_yaml_config() -> Dict[str, Any]:
  # Allow override via env if you want later
  config_path = os.getenv("CAUSALITYIQ_CONFIG", "config/config.yaml")
  path = Path(config_path)
  if not path.exists():
    raise FileNotFoundError(f"Config file not found: {path}")
  with path.open("r", encoding="utf-8") as f:
    return yaml.safe_load(f) or {}


def load_settings() -> Settings:
  raw = _load_yaml_config()
  return Settings(**raw)


# Singleton-style settings object
settings = load_settings()
