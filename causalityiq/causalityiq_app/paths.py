# src/paths.py

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List

from causalityiq_app.settings import settings


INDEX_FILE_NAME = "index.json"


# ---------- User ID / index handling ----------

def _normalize_email(email: str) -> str:
  """Normalize email for consistency (not for display)."""
  return email.strip().lower()


def _users_index_path() -> Path:
  """
  Path to the email ↔ user_id index file.
  Example: data/users/index.json
  """
  return settings.storage.base_data_dir / settings.storage.users_subdir / INDEX_FILE_NAME


def _load_users_index() -> Dict[str, str]:
  """
  Load the mapping {user_id: email_normalized}.
  Returns empty dict if file does not exist or is invalid.
  """
  path = _users_index_path()
  if not path.exists():
    return {}
  try:
    with path.open("r", encoding="utf-8") as f:
      data = json.load(f)
      if isinstance(data, dict):
        return data
  except json.JSONDecodeError:
    pass
  return {}


def _save_users_index(index: Dict[str, str]) -> None:
  """
  Save the mapping {user_id: email_normalized} to index.json.
  """
  path = _users_index_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    json.dump(index, f, indent=2, sort_keys=True)


def _make_user_id(email_normalized: str) -> str:
  """
  Create a stable user_id from the normalized email using SHA-256.
  Only the first 16 hex chars are used for brevity.
  """
  h = hashlib.sha256(email_normalized.encode("utf-8")).hexdigest()
  return f"u_{h[:16]}"


def get_or_create_user_id(email: str) -> str:
  """
  Given an email, return a stable user_id.
  If the email already exists in index.json, reuse its user_id.
  Otherwise, create a new entry and persist it.
  """
  email_norm = _normalize_email(email)
  index = _load_users_index()

  # Reuse existing mapping if present
  for user_id, stored_email in index.items():
    if stored_email == email_norm:
      return user_id

  # Create new user_id
  user_id = _make_user_id(email_norm)
  index[user_id] = email_norm
  _save_users_index(index)
  return user_id


# ---------- Base directories ----------

def users_base_dir() -> Path:
  """
  Base directory where all user directories live.
  Example: data/users
  """
  return settings.storage.users_base_dir


def user_base_dir_from_id(user_id: str) -> Path:
  """
  Root directory for a specific user.
  Example: data/users/u_9f3a7bc1c4d2e8a1
  """
  return users_base_dir() / user_id


def user_base_dir_from_email(email: str) -> Path:
  """
  Convenience helper: get/create user_id from email and return its base dir.
  """
  user_id = get_or_create_user_id(email)
  return user_base_dir_from_id(user_id)


# ---------- User-scoped directories ----------

def user_incidents_dir(user_id: str) -> Path:
  """
  Directory for all incidents belonging to a user.
  Example: data/users/<user_id>/incidents
  """
  return user_base_dir_from_id(user_id) / "incidents"


def user_workflows_dir(user_id: str) -> Path:
  """
  Directory for all workflow definitions belonging to a user.
  Example: data/users/<user_id>/workflows
  """
  return user_base_dir_from_id(user_id) / "workflows"


# ---------- Incident directory structure ----------

def incident_base_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Root directory for a specific incident under a profile.
  Example:
    data/users/<user_id>/incidents/rca/incident_001
  """
  return user_incidents_dir(user_id) / profile / incident_id


def incident_artifacts_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for all artifacts related to an incident.
  Example:
    .../incident_001/artifacts
  """
  return incident_base_dir(user_id, profile, incident_id) / "artifacts"


def incident_data_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for structured data (CSV, parquet, logs) for an incident.
  Example:
    .../incident_001/artifacts/data
  """
  return incident_artifacts_dir(user_id, profile, incident_id) / "data"


def incident_docs_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for documents (PDF, Word, Markdown) for an incident.
  Example:
    .../incident_001/artifacts/docs
  """
  return incident_artifacts_dir(user_id, profile, incident_id) / "docs"


def incident_images_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for images/charts/diagrams for an incident.
  Example:
    .../incident_001/artifacts/images
  """
  return incident_artifacts_dir(user_id, profile, incident_id) / "images"


def incident_analysis_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for analysis outputs (runs, intermediate data).
  Example:
    .../incident_001/analysis
  """
  return incident_base_dir(user_id, profile, incident_id) / "analysis"


def incident_reports_dir(user_id: str, profile: str, incident_id: str) -> Path:
  """
  Directory for generated reports for an incident.
  Example:
    .../incident_001/analysis/reports
  """
  return incident_analysis_dir(user_id, profile, incident_id) / "reports"


# ---------- Directory creation helpers ----------

def ensure_dir(path: Path) -> Path:
  """
  Ensure a directory exists, returning the path.
  """
  path.mkdir(parents=True, exist_ok=True)
  return path


def ensure_user_dirs(user_id: str) -> List[Path]:
  """
  Ensure the base directories for a user exist.
  Returns the list of created/ensured paths.
  """
  paths = [
    user_base_dir_from_id(user_id),
    user_incidents_dir(user_id),
    user_workflows_dir(user_id),
  ]
  return [ensure_dir(p) for p in paths]


def ensure_incident_dirs(user_id: str, profile: str, incident_id: str) -> List[Path]:
  """
  Ensure the full directory structure for an incident exists.
  Returns the list of created/ensured paths.
  """
  paths = [
    incident_base_dir(user_id, profile, incident_id),
    incident_artifacts_dir(user_id, profile, incident_id),
    incident_data_dir(user_id, profile, incident_id),
    incident_docs_dir(user_id, profile, incident_id),
    incident_images_dir(user_id, profile, incident_id),
    incident_analysis_dir(user_id, profile, incident_id),
    incident_reports_dir(user_id, profile, incident_id),
  ]
  return [ensure_dir(p) for p in paths]
