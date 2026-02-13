# causalityiq_app/reports.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from causalityiq_app.models.incident import Incident
from causalityiq_app.paths import incident_reports_dir
from causalityiq_app.analysis import basic_analysis_summary
from causalityiq_app.llm_client import generate_incident_analysis
from causalityiq_app.models.incident import Incident
from causalityiq_app.paths import incident_reports_dir
from causalityiq_app.data_summaries import (
    summarize_incident_signals,
    summarize_incident_documents,
)

def write_basic_markdown_report(
  user_id: str,
  profile: str,
  incident_id: str,
  incident: Incident,
) -> Path:
  """
  Write a simple Markdown summary for the incident into analysis/reports/.
  """
  reports_dir = incident_reports_dir(user_id, profile, incident_id)
  reports_dir.mkdir(parents=True, exist_ok=True)

  ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
  filename = f"{incident_id}_summary_{ts}.md"
  path = reports_dir / filename

  lines = []

  lines.append(f"# Incident: {incident.title}")
  lines.append("")
  lines.append(f"- ID: `{incident.id}`")
  lines.append(f"- Profile: `{incident.profile}`")
  lines.append(f"- Status: `{incident.status}`")

  if incident.created_at:
    lines.append(f"- Created at: `{incident.created_at.isoformat()}`")
  if incident.updated_at:
    lines.append(f"- Updated at: `{incident.updated_at.isoformat()}`")

  lines.append("")

  if incident.summary:
    lines.append("## Summary")
    lines.append("")
    lines.append(incident.summary)
    lines.append("")

  lines.append("## Context")
  lines.append("")
  ctx = incident.context
  if ctx.environment:
    lines.append(f"- Environment: `{ctx.environment}`")
  if ctx.service:
    lines.append(f"- Service: `{ctx.service}`")
  if ctx.region:
    lines.append(f"- Region: `{ctx.region}`")
  if ctx.tags:
    tags_str = ", ".join(ctx.tags)
    lines.append(f"- Tags: {tags_str}")
  lines.append("")

  if incident.hypotheses:
    lines.append("## Hypotheses")
    lines.append("")
    for h in incident.hypotheses:
      lines.append(f"- {h}")
    lines.append("")

  if incident.signals:
    lines.append("## Signals")
    lines.append("")
    for s in incident.signals:
      desc = f" ({s.description})" if s.description else ""
      unit = f" [{s.unit}]" if s.unit else ""
      path_info = f" – `{s.path}`" if s.path else ""
      lines.append(f"- **{s.name}**{unit}{desc}{path_info}")
    lines.append("")

  if incident.artifacts:
    lines.append("## Artifacts")
    lines.append("")
    if incident.artifacts.data_files:
      lines.append("- Data files:")
      for p in incident.artifacts.data_files:
        lines.append(f"  - `{p}`")
    if incident.artifacts.documents:
      lines.append("- Documents:")
      for p in incident.artifacts.documents:
        lines.append(f"  - `{p}`")
    if incident.artifacts.images:
      lines.append("- Images:")
      for p in incident.artifacts.images:
        lines.append(f"  - `{p}`")
    if incident.artifacts.links:
      lines.append("- Links:")
      for url in incident.artifacts.links:
        lines.append(f"  - {url}")
    lines.append("")

  # Basic non-LLM analysis
  lines.append("## Analysis summary")
  lines.append("")
  lines.append("```text")
  lines.append(basic_analysis_summary(incident))
  lines.append("```")
  lines.append("")


  content = "\n".join(lines)
  path.write_text(content, encoding="utf-8")
  return path

def append_llm_analysis_section(
  report_path: Path,
  incident: Incident,
  incident_base_dir: Path,  # NEW
) -> Path:
  """
  Append an 'LLM Analysis' section to an existing Markdown report.
  """
  summary_text = basic_analysis_summary(incident)

  # Build richer incident_text with optional problem_statement
  parts: list[str] = []

  problem_statement = getattr(incident, "problem_statement", None)
  if problem_statement:
    parts.append("Problem statement:")
    parts.append(problem_statement)

  parts.append("Incident summary:")
  parts.append(summary_text)

  # New: derived from files (goes only to the LLM prompt)
  parts.append(summarize_incident_signals(incident, incident_base_dir))
  parts.append(summarize_incident_documents(incident, incident_base_dir))
  
  incident_text = "\n\n".join(parts)

  prompts = incident.llm_prompts
  analysis_text = generate_incident_analysis(
    incident_text=incident_text,
    analysis_instructions=prompts.analysis_instructions,
    tone=prompts.tone,
    constraints=prompts.constraints,
  )

  existing = report_path.read_text(encoding="utf-8")
  lines = [existing.rstrip(), "", "## LLM Analysis", "", analysis_text, ""]
  report_path.write_text("\n".join(lines), encoding="utf-8")
  return report_path

