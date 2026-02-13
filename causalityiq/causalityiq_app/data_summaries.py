# causalityiq_app/data_summaries.py
from pathlib import Path
from typing import List
import pandas as pd
from docx import Document  # pip install python-docx

from causalityiq_app.models.incident import Incident  # adjust to your actual module

def summarize_signal_csv(csv_path: Path, signal_name: str, description: str | None = None) -> str:
    if not csv_path.exists():
        return f"- {signal_name}: CSV file not found at {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"- {signal_name}: failed to read CSV ({e})"

    if df.empty:
        return f"- {signal_name}: no data rows."

    # Assume first column is time, rest numeric
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    parts: List[str] = [f"- {signal_name}: {description or ''}".strip()]

    for col in numeric_cols[:3]:  # cap at a few cols for brevity
        series = df[col].dropna()
        if series.empty:
            continue
        parts.append(
            f"  - {col}: min={series.min():.3f}, max={series.max():.3f}, "
            f"mean={series.mean():.3f}, count={series.count()}"
        )

    if len(parts) == 1:
        parts.append("  - No numeric columns with data.")
    return "\n".join(parts)


def summarize_incident_signals(incident: Incident, base_dir: Path) -> str:
    if not incident.signals:
        return "Key findings from time series:\n- No time series signals configured."

    summaries: List[str] = []
    for sig in incident.signals:
        csv_path = base_dir / sig.path  # e.g. artifacts/data/...
        summaries.append(
            summarize_signal_csv(csv_path, sig.name, sig.description)
        )

    return "Key findings from time series:\n" + "\n".join(summaries)

from docx import Document  # pip install python-docx

def summarize_incident_documents(incident: Incident, base_dir: Path, max_chars: int = 1200) -> str:
    docs = (incident.artifacts.documents or []) if incident.artifacts else []
    if not docs:
        return "Relevant document excerpts:\n- No documents attached."

    excerpts: List[str] = []
    for rel_path in docs:
        path = base_dir / rel_path
        if not path.exists():
            excerpts.append(f"- {rel_path}: file not found.")
            continue

        text = ""
        if path.suffix.lower() == ".docx":
            try:
                doc = Document(path)
                text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except Exception as e:
                excerpts.append(f"- {rel_path}: failed to parse ({e}).")
                continue
        else:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                excerpts.append(f"- {rel_path}: failed to read ({e}).")
                continue

        text = text.strip()
        if not text:
            excerpts.append(f"- {rel_path}: document is empty.")
            continue

        if len(text) > max_chars:
            text = text[:max_chars] + " [...]"

        excerpts.append(f"- {rel_path} excerpt:\n{text}")

    return "Relevant document excerpts:\n" + "\n\n".join(excerpts)
