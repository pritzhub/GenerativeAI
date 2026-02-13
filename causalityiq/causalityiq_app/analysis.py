# causalityiq_app/analysis.py
from __future__ import annotations

from causalityiq_app.models.incident import Incident

def basic_analysis_summary(incident: Incident) -> str:
  """
  Return a short, human-readable summary of what's configured so far.
  No LLM, just counts and simple checks.
  """
  parts: list[str] = []

  parts.append(f"Incident '{incident.id}' ({incident.title}) analysis summary:")

  # Signals overview
  if incident.signals:
    parts.append(f"- Signals configured: {len(incident.signals)}")
    names = ", ".join(s.name for s in incident.signals)
    parts.append(f"  - Names: {names}")
  else:
    parts.append("- Signals configured: 0 (no time-series/metrics attached yet)")

  # Artifacts overview
  a = incident.artifacts
  total_artifacts = len(a.data_files) + len(a.documents) + len(a.images) + len(a.links)
  parts.append(f"- Artifacts total: {total_artifacts}")
  if a.data_files:
    parts.append(f"  - Data files: {len(a.data_files)}")
  if a.documents:
    parts.append(f"  - Documents: {len(a.documents)}")
  if a.images:
    parts.append(f"  - Images: {len(a.images)}")
  if a.links:
    parts.append(f"  - Links: {len(a.links)}")

  # Hypotheses overview
  if incident.hypotheses:
    parts.append(f"- Hypotheses: {len(incident.hypotheses)}")
  else:
    parts.append("- Hypotheses: none captured yet")

  return "\n".join(parts)
