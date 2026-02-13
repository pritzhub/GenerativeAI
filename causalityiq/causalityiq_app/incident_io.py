from __future__ import annotations

from pathlib import Path
from typing import Optional

from causalityiq_app.models.incident import Incident
from causalityiq_app.paths import incident_base_dir


def incident_json_path(user_id: str, profile: str, incident_id: str) -> Path:
  return incident_base_dir(user_id, profile, incident_id) / "incident.json"


def load_incident(user_id: str, profile: str, incident_id: str) -> Optional[Incident]:
  path = incident_json_path(user_id, profile, incident_id)
  if not path.exists():
    return None
  data = path.read_text(encoding="utf-8")
  return Incident.model_validate_json(data)


def save_incident(user_id: str, profile: str, incident: Incident) -> Path:
  path = incident_json_path(user_id, profile, incident.id)
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(incident.model_dump_json(indent=2), encoding="utf-8")
  return path
