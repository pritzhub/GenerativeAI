# causalityiq_ui/streamlit_app.py
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st

# Adjust this path if your backend repo is elsewhere
BACKEND_ROOT = Path("../causalityiq").resolve()
DATA_ROOT = BACKEND_ROOT / "data"
USERS_DIR = DATA_ROOT / "users"


# ---------- Helpers: paths, load/save ----------

def user_hash_from_email(email: str) -> str:
    return "u_" + hashlib.sha1(email.strip().lower().encode("utf-8")).hexdigest()[:16]


def incident_dir(email: str, profile: str, incident_id: str) -> Path:
    return USERS_DIR / user_hash_from_email(email) / "incidents" / profile / incident_id


def incident_json_path(email: str, profile: str, incident_id: str) -> Path:
    return incident_dir(email, profile, incident_id) / "incident.json"


def load_incident(email: str, profile: str, incident_id: str) -> Dict[str, Any] | None:
    path = incident_json_path(email, profile, incident_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_incident(email: str, profile: str, data: Dict[str, Any]) -> Path:
    inc_dir = incident_dir(email, profile, data["id"])
    inc_dir.mkdir(parents=True, exist_ok=True)
    path = inc_dir / "incident.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def list_incidents(email: str, profile: str) -> List[str]:
    u_hash = user_hash_from_email(email)
    base = USERS_DIR / u_hash / "incidents" / profile

    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


# ---------- Streamlit app ----------

def main():
    st.title("CausalityIQ – RCA UI (interim)")

    # Identity & profile
    email = st.text_input("Your email", value="")  # no default
    profile = st.selectbox("Profile", ["rca", "general"], index=0)

    # Incident selection (depends on email + profile)
    existing_ids: List[str] = []
    if email.strip():
        existing_ids = list_incidents(email, profile)
            
    selected = st.selectbox(
        "Incident ID",
        options=(["<new>"] + existing_ids) if email.strip() else ["<new>"],
        index=0,
        help="Select an existing incident or choose <new> to create one.",
    )

    if selected == "<new>":
        default_new_id = "incident_001"
        incident_id = st.text_input("New incident ID", value=default_new_id)
        existing = None
    else:
        incident_id = selected
        #path = incident_json_path(email, profile, incident_id)
        existing = load_incident(email, profile, incident_id)

        
    st.markdown("---")
    st.subheader("Incident metadata")

    title = st.text_input("Incident title", value=(existing or {}).get("title", ""))
    summary = st.text_input("Short summary", value=(existing or {}).get("summary", ""))

    problem_statement = st.text_area(
        "Problem statement",
        value=(existing or {}).get("problem_statement", ""),
        height=150,
        help="Describe the observed issue, time window, and impact.",
    )

    ctx = (existing or {}).get("context", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        environment = st.text_input("Environment", value=ctx.get("environment", "prod"))
    with col2:
        service = st.text_input("Service / System", value=ctx.get("service", "PPC"))
    with col3:
        region = st.text_input("Region", value=ctx.get("region", "KSA"))

    st.subheader("LLM prompts")

    llm = (existing or {}).get("llm_prompts", {})
    analysis_instructions = st.text_area(
        "Analysis instructions",
        value=llm.get(
            "analysis_instructions",
            "This incident is for Power Plant Controller (PPC). "
            "Focus on correlating PQM measurement, PPC Master & slaves core input and output, "
            "and error codes, and explain likely cause of the problem.",
        ),
        height=120,
    )
    tone = st.text_input(
        "Tone",
        value=llm.get(
            "tone",
            "Formal, suitable for a customer-facing incident report.",
        ),
    )
    constraints = st.text_area(
        "Constraints",
        value=llm.get(
            "constraints",
            "Avoid internal hostnames and customer names. Limit main narrative to ~600 words.",
        ),
        height=80,
    )

    # Build incident_data (JSON) using existing as base
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    base = existing or {}
    ctx_tags = ctx.get("tags", [])
    ctx_extra = ctx.get("extra", {})

    incident_data: Dict[str, Any] = {
        "id": incident_id,
        "profile": profile,
        "title": title,
        "summary": summary,
        "problem_statement": problem_statement,
        "created_at": base.get("created_at", now_iso),
        "updated_at": now_iso,
        "status": base.get("status", "draft"),
        "context": {
            "environment": environment,
            "service": service,
            "region": region,
            "tags": ctx_tags,
            "extra": ctx_extra,
        },
        "time_window": base.get("time_window", {"start": None, "end": None}),
        "signals": base.get("signals", []),
        "artifacts": base.get(
            "artifacts",
            {"data_files": [], "documents": [], "images": [], "links": []},
        ),
        "hypotheses": base.get("hypotheses", []),
        "report_template": base.get("report_template", "default_report"),
        "llm_prompts": {
            "analysis_instructions": analysis_instructions,
            "tone": tone,
            "constraints": constraints,
            "extra": llm.get("extra", {}),
        },
        "labels": base.get("labels", []),
        "metadata": base.get(
            "metadata",
            {
                "created_by": email,
                "severity": None,
                "customer_impact": None,
                "extra": {},
            },
        ),
    }

    st.markdown("---")
    st.subheader("Artifacts upload")

    inc_dir = incident_dir(email, profile, incident_id)
    artifacts_dir = inc_dir / "artifacts"
    data_dir = artifacts_dir / "data"
    docs_dir = artifacts_dir / "docs"
    images_dir = artifacts_dir / "images"
    data_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Data files (CSV/TXT)
    data_files = st.file_uploader(
        "Upload CSV/TXT data files",
        type=["csv", "txt"],
        accept_multiple_files=True,
    )
    for f in data_files:
        save_path = data_dir / f.name
        save_path.write_bytes(f.read())
        rel = str(save_path.relative_to(inc_dir))
        if rel not in incident_data["artifacts"]["data_files"]:
            incident_data["artifacts"]["data_files"].append(rel)

    # Documents
    docs = st.file_uploader(
        "Upload documents",
        type=["docx", "pdf", "txt"],
        accept_multiple_files=True,
    )
    for f in docs:
        save_path = docs_dir / f.name
        save_path.write_bytes(f.read())
        rel = str(save_path.relative_to(inc_dir))
        if rel not in incident_data["artifacts"]["documents"]:
            incident_data["artifacts"]["documents"].append(rel)

    # Images
    imgs = st.file_uploader(
        "Upload images",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )
    for f in imgs:
        save_path = images_dir / f.name
        save_path.write_bytes(f.read())
        rel = str(save_path.relative_to(inc_dir))
        if rel not in incident_data["artifacts"]["images"]:
            incident_data["artifacts"]["images"].append(rel)

    # Quick view of current artifacts
    st.write("Current data files:", incident_data["artifacts"]["data_files"])
    st.write("Current documents:", incident_data["artifacts"]["documents"])
    st.write("Current images:", incident_data["artifacts"]["images"])

    st.markdown("---")

    if st.button("Save incident"):
        save_incident(email, profile, incident_data)
        st.success("Incident saved.")

    if st.button("Run analysis and view report"):
        # Save latest JSON
        save_incident(email, profile, incident_data)

        # Call backend app via CLI (adjust if your app.py needs arguments)
        BACKEND_PYTHON = str((BACKEND_ROOT / ".venv" / "Scripts" / "python.exe").resolve())
        
        #cmd = [
        #    sys.executable,
        #    "-m",
        #    "causalityiq_app.app",
        #]
        #cmd = [
        #    BACKEND_PYTHON,
        #    "-m",
        #    "causalityiq_app.app",
        #]
        cmd = [
            BACKEND_PYTHON,
            "-m",
            "causalityiq_app.app",
            "--non-interactive",
            "--email", email,
            "--profile", profile,
            "--incident-id", incident_id,
        ]
        subprocess.run(cmd, cwd=BACKEND_ROOT, check=True)

        try:
            subprocess.run(cmd, cwd=BACKEND_ROOT, check=True)
        except subprocess.CalledProcessError as e:
            st.error(f"Backend run failed: {e}")
        else:
            # Find latest report for this incident
            reports_dir = inc_dir / "analysis" / "reports"
            if reports_dir.exists():
                reports = sorted(reports_dir.glob("*.md"))
                if reports:
                    latest = reports[-1]
                    st.success(f"Analysis complete. Report: {latest}")
                    st.markdown("### Report")
                    st.markdown(latest.read_text(encoding="utf-8"))
                else:
                    st.warning("No report files found yet.")
            else:
                st.warning("Reports directory does not exist yet.")


if __name__ == "__main__":
    main()
