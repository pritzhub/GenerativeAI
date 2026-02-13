# causalityiq_app/langgraph_flow.py
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, START, END  # [web:149][web:152]

from causalityiq_app.incident_io import load_incident
from causalityiq_app.analysis import basic_analysis_summary
from causalityiq_app.reports import write_basic_markdown_report
from pathlib import Path
from causalityiq_app.reports import append_llm_analysis_section

class GraphState(TypedDict):
  user_id: str
  profile: str
  incident_id: str
  summary_text: str | None
  report_path: str | None


def load_incident_node(state: GraphState) -> GraphState:
  """
  Node 1: load Incident from disk and log a simple summary into state.
  """
  incident = load_incident(state["user_id"], state["profile"], state["incident_id"])
  if incident is None:
    raise RuntimeError(
      f"incident.json not found for user_id={state['user_id']}, "
      f"profile={state['profile']}, incident_id={state['incident_id']}"
    )

  summary = basic_analysis_summary(incident)
  state["summary_text"] = summary
  return state

def write_report_node(state: GraphState) -> GraphState:
  incident = load_incident(state["user_id"], state["profile"], state["incident_id"])
  if incident is None:
    raise RuntimeError(
      f"incident.json not found when writing report for incident_id={state['incident_id']}"
    )

  path = write_basic_markdown_report(
    user_id=state["user_id"],
    profile=state["profile"],
    incident_id=state["incident_id"],
    incident=incident,
  )
  state["report_path"] = str(path)
  return state

def build_graph():
  workflow = StateGraph(GraphState)

  workflow.add_node("load_incident", load_incident_node)
  workflow.add_node("write_report", write_report_node)
  workflow.add_node("llm_analysis", llm_analysis_node)

  workflow.add_edge(START, "load_incident")
  workflow.add_edge("load_incident", "write_report")
  workflow.add_edge("write_report", "llm_analysis")
  workflow.add_edge("llm_analysis", END)

  app = workflow.compile()
  return app
  
def llm_analysis_node(state: GraphState) -> GraphState:
  """
  Node 3: call LLM using llm_prompts and append an LLM Analysis section.
  """
  incident = load_incident(state["user_id"], state["profile"], state["incident_id"])
  if incident is None:
    raise RuntimeError(
      f"incident.json not found when running LLM analysis for incident_id={state['incident_id']}"
    )

  if not state.get("report_path"):
    raise RuntimeError("report_path is missing in state; run write_report_node first.")

  report_path = Path(state["report_path"])
  if not report_path.exists():
    raise RuntimeError(f"Report file does not exist: {report_path}")

  # Derive base dir: parent of 'analysis/reports/...'
  incident_base_dir = report_path.parent.parent.parent  # adjust if your layout differs
    
  append_llm_analysis_section(report_path, incident, incident_base_dir)
  return state