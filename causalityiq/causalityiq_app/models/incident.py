from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from pydantic import BaseModel, Field


class TimeWindow(BaseModel):
  start: Optional[datetime] = None
  end: Optional[datetime] = None


class Signal(BaseModel):
  name: str
  description: Optional[str] = None
  unit: Optional[str] = None
  source_type: Optional[str] = Field(
    default=None,
    description="e.g. timeseries_csv, metrics_api, log_file"
  )
  path: Optional[str] = Field(
    default=None,
    description="Relative path to data under incident directory"
  )

class Artifacts(BaseModel):
  data_files: List[str] = Field(default_factory=list)
  documents: List[str] = Field(default_factory=list)
  images: List[str] = Field(default_factory=list)
  links: List[str] = Field(default_factory=list)

class Context(BaseModel):
  environment: Optional[str] = None
  service: Optional[str] = None
  region: Optional[str] = None
  tags: List[str] = Field(default_factory=list)
  extra: Dict[str, str] = Field(default_factory=dict)

class IncidentMetadata(BaseModel):
  created_by: Optional[str] = None
  severity: Optional[str] = None
  customer_impact: Optional[str] = None
  extra: Dict[str, str] = Field(default_factory=dict)

class LLMPrompts(BaseModel):
  analysis_instructions: str | None = Field(
    default=None,
    description="High-level instructions for how the model should analyze this incident.",
  )
  tone: str | None = Field(
    default=None,
    description="Desired tone/style, e.g. 'formal customer-facing', 'internal technical'.",
  )
  constraints: str | None = Field(
    default=None,
    description="Constraints such as anonymization, length limits, or compliance notes.",
  )
  extra: Dict[str, str] = Field(
    default_factory=dict,
    description="Any additional prompt fragments or flags.",
  )

class Incident(BaseModel):
  id: str
  profile: str = Field(
    default="generic",
    description="Logical use-case profile (rca, postmortem, capacity, etc.)"
  )
  title: str
  summary: Optional[str] = None
  # NEW:
  problem_statement: Optional[str] = None
  created_at: Optional[datetime] = None
  updated_at: Optional[datetime] = None
  status: str = Field(default="draft")

  context: Context = Field(default_factory=Context)
  time_window: TimeWindow = Field(default_factory=TimeWindow)

  signals: List[Signal] = Field(default_factory=list)
  artifacts: Artifacts = Field(default_factory=Artifacts)

  hypotheses: List[str] = Field(default_factory=list)
   
  report_template: str = Field(
    default="default_report",
    description="Key into templates section of config"
  )

  llm_prompts: LLMPrompts = Field(
    default_factory=LLMPrompts,
    description="Optional LLM prompt configuration for this incident."
  )
  
  labels: List[str] = Field(default_factory=list)
  metadata: IncidentMetadata = Field(default_factory=IncidentMetadata)

  class Config:
    # Ensure JSON serialization is friendly
    json_encoders = {
      Path: lambda p: str(p),
    }
