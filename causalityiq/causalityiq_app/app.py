# src/app/app.py
from __future__ import annotations

import logging
from datetime import datetime, timezone


from causalityiq_app.logging_config import setup_logging
from causalityiq_app.paths import (
  get_or_create_user_id,
  ensure_user_dirs,
  ensure_incident_dirs,
)
from causalityiq_app.models.incident import Incident, Context, TimeWindow, Artifacts
from causalityiq_app.incident_io import save_incident, load_incident
from causalityiq_app.reports import write_basic_markdown_report
from causalityiq_app.langgraph_flow import build_graph
import argparse

logger = logging.getLogger(__name__)


def run_app() -> None:
  # 1) Setup logging (console + rotating file)
  setup_logging()
  logger.info("Starting CausalityIQ app")

  # 2) Ask for user email and resolve user_id
  email = input("Enter your email (for saving your work): ").strip()
  if not email:
    print("Email is required to continue.")
    logger.error("No email provided, exiting.")
    return

  user_id = get_or_create_user_id(email)
  logger.info("Using user_id=%s for email=%s", user_id, email)

  # 3) Ensure base user directories exist
  ensure_user_dirs(user_id)
  logger.info("Ensured base directories for user_id=%s", user_id)

  # 4) For now, ask for a simple profile and incident_id
  profile = input("Enter analysis profile (e.g. rca, generic): ").strip() or "generic"
  incident_id = input("Enter incident ID (e.g. incident_001): ").strip() or "incident_001"

  ensure_incident_dirs(user_id, profile, incident_id)
  logger.info(
    "Ensured directories for user_id=%s, profile=%s, incident_id=%s",
    user_id,
    profile,
    incident_id,
  )
  
  existing_incident = load_incident(user_id, profile, incident_id)
  if existing_incident:
    logger.info(
      "Existing incident.json found for user_id=%s, profile=%s, incident_id=%s",
      user_id,
      profile,
      incident_id,
    )
  else:
    logger.info(
      "No existing incident.json for user_id=%s, profile=%s, incident_id=%s",
      user_id,
      profile,
      incident_id,
    )

  # --- New: collect minimal incident info and create stub incident.json ---
  print("\nLet's create or update incident.json for this incident.")

  # Helper to build prompts with defaults
  def ask(prompt: str, default: str | None) -> str | None:
    if default:
      full_prompt = f"{prompt} [{default}]: "
    else:
      full_prompt = f"{prompt}: "
    value = input(full_prompt).strip()
    if value:
      return value
    return default

  # Derive defaults from existing incident if present
  default_title = existing_incident.title if existing_incident else f"Incident {incident_id}"
  default_summary = existing_incident.summary if existing_incident else None
  default_env = existing_incident.context.environment if existing_incident else None
  default_service = existing_incident.context.service if existing_incident else None
  default_region = existing_incident.context.region if existing_incident else None

  title = ask("Incident title", default_title)
  summary = ask("Short summary (optional)", default_summary)
  environment = ask("Environment (e.g. prod, dev)", default_env)
  service = ask("Service/system name", default_service)
  region = ask("Region", default_region)

  now = datetime.now(timezone.utc)

  if existing_incident:
    incident = existing_incident.model_copy(update={
      "title": title,
      "summary": summary,
      "updated_at": now,
      "context": existing_incident.context.model_copy(update={
        "environment": environment,
        "service": service,
        "region": region,
      }),
    })
  else:
    incident = Incident(
      id=incident_id,
      profile=profile,
      title=title or f"Incident {incident_id}",
      summary=summary,
      created_at=now,
      updated_at=now,
      status="draft",
      context=Context(
        environment=environment,
        service=service,
        region=region,
      ),
      time_window=TimeWindow(),
      artifacts=Artifacts(),
      report_template="default_report",
      metadata={"created_by": email},
    )

  path = save_incident(user_id, profile, incident)
  logger.info("Created stub incident.json at %s", path)

  print("\nCausalityIQ setup complete.")
  print(f"- user_id: {user_id}")
  print(f"- profile: {profile}")
  print(f"- incident_id: {incident_id}")
  print(f"- incident.json: {path}")

  # Quick sanity check: reload incident.json and print title
  # Load back and write a basic Markdown report

  loaded = load_incident(user_id, profile, incident_id)
  if loaded:
    logger.info(
      "Loaded incident back from disk: id=%s, title=%s",
      loaded.id,
      loaded.title,
    )
    report_path = write_basic_markdown_report(
      user_id=user_id,
      profile=profile,
      incident_id=incident_id,
      incident=loaded,
    )
    logger.info("Wrote basic report to %s", report_path)
    print(f"\nLoaded incident back from disk: {loaded.id} – {loaded.title}")
    print(f"Basic report written to: {report_path}")
  else:
    logger.error("Failed to reload incident.json for %s", incident_id)

  # NEW: ask to run LangGraph flow
  run_graph = input("\nRun LangGraph flow for this incident? [y/N]: ").strip().lower()
  if run_graph == "y":
    graph_app = build_graph()
    result_state = graph_app.invoke({
      "user_id": user_id,
      "profile": profile,
      "incident_id": incident_id,
      "summary_text": None,
      "report_path": None,
    })
    logger.info("LangGraph run completed with state: %s", result_state)
    print("LangGraph flow executed (load_incident -> write_report -> llm_analysis).")

  print("You can now add data/docs/images under this incident and extend the JSON as needed.")

def run_app_with_params(
    email: str,
    profile: str,
    incident_id: str,
    non_interactive: bool = False,
) -> None:
    setup_logging()
    logger.info("Starting CausalityIQ app")

    if not email:
        print("Email is required to continue.")
        logger.error("No email provided, exiting.")
        return

    # 1) Resolve user and incident paths
    user_id = get_or_create_user_id(email)
    logger.info("Using user_id=%s for email=%s", user_id, email)

    ensure_user_dirs(user_id)
    logger.info("Ensured base directories for user_id=%s", user_id)

    ensure_incident_dirs(user_id, profile, incident_id)
    logger.info(
        "Ensured directories for user_id=%s, profile=%s, incident_id=%s",
        user_id,
        profile,
        incident_id,
    )

    # 2) Load existing incident.json (if any)
    existing_incident = load_incident(user_id, profile, incident_id)
    if existing_incident:
        logger.info(
            "Existing incident.json found for user_id=%s, profile=%s, incident_id=%s",
            user_id,
            profile,
            incident_id,
        )
    else:
        logger.info(
            "No existing incident.json for user_id=%s, profile=%s, incident_id=%s",
            user_id,
            profile,
            incident_id,
        )

    print("\nLet's create or update incident.json for this incident.")

    def ask(prompt: str, default: str | None) -> str | None:
        if non_interactive:
            # UI mode: never prompt, just use defaults from existing_incident
            return default
        if default:
            full_prompt = f"{prompt} [{default}]: "
        else:
            full_prompt = f"{prompt}: "
        value = input(full_prompt).strip()
        if value:
            return value
        return default

    # 3) Derive defaults from existing incident
    default_title = existing_incident.title if existing_incident else f"Incident {incident_id}"
    default_summary = existing_incident.summary if existing_incident else None
    default_env = existing_incident.context.environment if existing_incident else None
    default_service = existing_incident.context.service if existing_incident else None
    default_region = existing_incident.context.region if existing_incident else None

    # 4) Collect / reuse high‑level incident metadata
    title = ask("Incident title", default_title)
    summary = ask("Short summary (optional)", default_summary)
    environment = ask("Environment (e.g. prod, dev)", default_env)
    service = ask("Service/system name", default_service)
    region = ask("Region", default_region)

    now = datetime.now(timezone.utc)

    if existing_incident:
        incident = existing_incident.model_copy(update={
            "title": title,
            "summary": summary,
            "updated_at": now,
            "context": existing_incident.context.model_copy(update={
                "environment": environment,
                "service": service,
                "region": region,
            }),
        })
    else:
        incident = Incident(
            id=incident_id,
            profile=profile,
            title=title or f"Incident {incident_id}",
            summary=summary,
            created_at=now,
            updated_at=now,
            status="draft",
            context=Context(
                environment=environment,
                service=service,
                region=region,
            ),
            time_window=TimeWindow(),
            artifacts=Artifacts(),
            report_template="default_report",
            metadata={"created_by": email},
        )

    # 5) Save incident.json
    path = save_incident(user_id, profile, incident)
    logger.info("Created/updated incident.json at %s", path)

    print("\nCausalityIQ setup complete.")
    print(f"- user_id: {user_id}")
    print(f"- profile: {profile}")
    print(f"- incident_id: {incident_id}")
    print(f"- incident.json: {path}")

    # 6) Reload incident and write basic markdown report skeleton
    loaded = load_incident(user_id, profile, incident_id)
    if loaded:
        logger.info(
            "Loaded incident back from disk: id=%s, title=%s",
            loaded.id,
            loaded.title,
        )
        # IMPORTANT: this should write to your canonical report path,
        # e.g. data/users/<user_id>/incidents/<profile>/<incident_id>/reports/report.md
        report_path = write_basic_markdown_report(
            user_id=user_id,
            profile=profile,
            incident_id=incident_id,
            incident=loaded,
        )
        logger.info("Wrote basic report skeleton to %s", report_path)
        print(f"\nLoaded incident back from disk: {loaded.id} – {loaded.title}")
        print(f"Basic report skeleton written to: {report_path}")
    else:
        logger.error("Failed to reload incident.json for %s", incident_id)
        # If we cannot load, no point running LangGraph
        return

    # 7) Run LangGraph to produce the final LLM‑enhanced report
    if non_interactive:
        # UI path: always run full analysis, no prompt
        graph_app = build_graph()
        result_state = graph_app.invoke({
            "user_id": user_id,
            "profile": profile,
            "incident_id": incident_id,
            "summary_text": None,
            "report_path": str(report_path),
        })
        logger.info("LangGraph run completed with state: %s", result_state)
    else:
        # CLI path: ask user whether to run LLM analysis
        run_graph = input("\nRun LangGraph flow for this incident? [y/N]: ").strip().lower()
        if run_graph == "y":
            graph_app = build_graph()
            result_state = graph_app.invoke({
                "user_id": user_id,
                "profile": profile,
                "incident_id": incident_id,
                "summary_text": None,
                "report_path": str(report_path),
            })
            logger.info("LangGraph run completed with state: %s", result_state)
            print("LangGraph flow executed (load_incident -> write_report -> llm_analysis).")

    print("You can now add data/docs/images under this incident and extend the JSON as needed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--email")
    parser.add_argument("--profile")
    parser.add_argument("--incident-id")
    parser.add_argument("--non-interactive", action="store_true")
    args = parser.parse_args()

    if args.non_interactive:            # UI
        if not (args.email and args.profile and args.incident_id):
            raise SystemExit("Missing --email/--profile/--incident-id in non-interactive mode")
        run_app_with_params(
            email=args.email,
            profile=args.profile,
            incident_id=args.incident_id,
            non_interactive=True,
        )
    else:
        run_app()       # Command prompt

if __name__ == "__main__":
  main()