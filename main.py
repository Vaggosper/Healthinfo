# main.py
import os, re, time, json
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd
from openai import OpenAI

# ---------- PAGE ----------
st.set_page_config(page_title="Health Insight — OpenAI-only", page_icon="🩺", layout="wide")

# ---------- OPENAI KEY ----------
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("Λείπει το OpenAI API key. Πρόσθεσέ το στα Secrets ως OPENAI_API_KEY.")
    st.stop()
client = OpenAI(api_key=API_KEY)

MODEL_NAME = "gpt-4o-mini"  # γρήγορο/φθηνό

# ---------- PROMPTS ----------
SYSTEM_INSTRUCTIONS = """
You are a careful medical information formatter. You NEVER give medical advice.
You ONLY return JSON that fits the schema. If you don't know something, estimate conservatively.
Percentages must be strings with a percent sign (e.g., "72.4%"). Integers must be integers.
"""

USER_TEMPLATE = """
Provide structured, didactic information about the disease: "{disease}".
Return STRICT JSON (no prose outside JSON) with the following schema:

{
  "name": string,
  "summary": string,
  "statistics": {
    "total_cases": integer,
    "incidence_per_100k": number,
    "recovery_rate": string,
    "mortality_rate": string
  },
  "region_breakdown": [
    {"region": string, "cases": integer, "deaths": integer}
  ],
  "recovery_options": {
    "<option_name>": "1-3 plain sentences (no medical advice, general info)"
  },
  "medications": [
    {"name": string, "side_effects": [string, ...], "dosage": string}
  ],
  "disclaimer": "This content is educational only and not medical advice."
}

Rules:
- Output MUST be valid JSON with double quotes only. No markdown, no backticks, no text outside JSON.
"""

# ---------- HELPERS ----------
def coerce_pct(s: Any) -> float:
    try:
        return float(str(s).strip().replace("%", "").replace(",", "."))
    except Exception:
        return 0.0

def extract_json_block(text: str) -> str:
    """Πάρε το πρώτο {...} block ακόμη κι αν έχει κείμενο γύρω-γύρω."""
    if not isinstance(text, str):
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else ""

def safe_load_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        block = extract_json_block(text)
        if block:
            try:
                return json.loads(block)
            except Exception:
                pass
    return {}

@st.cache_data(show_spinner=False, ttl=600)
def call_openai(disease: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Χρήση Responses API με input ως ΕΝΑ string (σωστός τρόπος).
    Επιστρέφει (ok, data, raw_or_error).
    """
    last = ""
    user_text = USER_TEMPLATE.format(disease=disease)
    # Ενώνουμε system + user σε ένα καθαρό prompt-string
    prompt = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + user_text.strip()

    for _ in range(3):
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_output_tokens=1200,
                input=prompt,  # <-- ΣΗΜΑΝΤΙΚΟ: string, ΟΧΙ messages
            )
            raw = resp.output_text  # string
            data = safe_load_json(raw)
            if data:
                return True, data, raw
            last = "Invalid JSON from model"
        except Exception as ex:
            last = f"{type(ex).__name__}: {ex}"
        time.sleep(1.0)
    return False, {}, last

def render_stats(info: Dict[str, Any]):
    stats = info.get("statistics", {}) or {}
    rec = coerce_pct(stats.get("recovery_rate", "0%"))
    mort = coerce_pct(stats.get("mortality_rate", "0%"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recovery rate", stats.get("recovery_rate", "—"))
    c2.metric("Mortality rate", stats.get("mortality_rate", "—"))
    c3.metric("Total cases", f"{stats.get('total_cases', 0):,}")
    c4.metric("Incidence / 100k", stats.get("incidence_per_100k", "—"))
    df = pd.DataFrame({"Rate": ["Recovery", "Mortality"], "Value": [rec, mort]}).set_index("Rate")
    st.bar_chart(df)

def render_regions(info: Dict[str, Any]):
    rows: List[Dict[str, Any]] = info.get("region_breakdown", []) or []
    if not rows:
        return
    df = pd.DataFrame(rows)
    st.subheader("Regional breakdown")
    st.dataframe(df, use_container_width=True)
    try:
        st.bar_chart(df.set_index("region")["cases"])
    except Exception:
        pass

def render_options(info: Dict[str, Any]):
    opts: Dict[str, str] = info.get("recovery_options", {}) or {}
    if not opts:
        return
    st.subheader("Recovery options (general info)")
    for k, v in opts.items():
        st.markdown(f"**{k}**")
        st.write(v)

def render_meds(info: Dict[str, Any]):
    meds: List[Dict[str, Any]] = info.get("medications", []) or []
    if not meds:
        return
    st.subheader("Medications (examples)")
    for i, m in enumerate(meds, start=1):
        name = (m or {}).get("name", "—")
        dose = (m or {}).get("dosage", "—")
        se = (m or {}).get("side_effects", []) or []
        st.markdown(f"**{i}. {name}**")
        st.write(f"Dosage: {dose}")
        if se:
            st.write("Side effects:")
            for s in se:
                st.write(f"· {s}")

# ---------- UI ----------
st.title("🩺 Health Insight — OpenAI-only")
st.caption("Εκπαιδευτικό εργαλείο. Δεν παρέχει ιατρικές συμβουλές. Χωρίς εξωτερικά APIs (μόνο OpenAI).")

disease = st.text_input("Πληκτρολόγησε ασθένεια (π.χ. influenza, diabetes, malaria):", "")
if st.button("Ανάλυση") and disease.strip():
    with st.spinner("Φορτώνω…"):
        ok, data, raw = call_openai(disease.strip())

    if ok:
        try:
            st.success("ΟΚ — λήψη δεδομένων.")
            st.header(data.get("name", disease.strip()))
            if data.get("summary"):
                st.write(data.get("summary"))
            render_stats(data)
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                render_regions(data)
            with c2:
                render_options(data)
            st.divider()
            render_meds(data)
            if data.get("disclaimer"):
                st.info(data.get("disclaimer"))
            with st.expander("Raw JSON (debug)"):
                st.code(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            st.error("Σφάλμα κατά την εμφάνιση των δεδομένων.")
            with st.expander("Debug"):
                st.write(repr(e))
                try:
                    st.code(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    st.write(data)
    else:
        st.error("Αποτυχία κλήσης στο OpenAI.")
        with st.expander("Debug details"):
            st.write(raw if isinstance(raw, str) else repr(raw))
else:
    st.write("👆 Γράψε μια ασθένεια και πάτα *Ανάλυση* για να ξεκινήσουμε.")

