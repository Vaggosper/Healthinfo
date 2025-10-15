# main.py
import os
import time
import json
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI


# ---------------------------
# PAGE / THEME
# ---------------------------
st.set_page_config(
    page_title="Health Insight — OpenAI-only",
    page_icon="🩺",
    layout="wide",
)

# ---------------------------
# OPENAI KEY (secrets + env)
# ---------------------------
api_key = (
    st.secrets.get("OPENAI_API_KEY")      # Streamlit Cloud secrets UI
    or os.environ.get("OPENAI_API_KEY")   # ENV fallback
)

if not api_key:
    st.error("Λείπει το OpenAI API key. Πρόσθεσέ το στα Secrets ως **OPENAI_API_KEY**.")
    st.stop()

client = OpenAI(api_key=api_key)

# ---------------------------
# PROMPTS
# ---------------------------
SYSTEM_INSTRUCTIONS = """
You are a careful medical information formatter. You NEVER give medical advice.
You ONLY return JSON that fits the schema. If you don't know something, estimate clearly and keep values plausible.
Percentages must be strings with a percent sign (e.g., "72.4%"). Integers must be integers.
"""

USER_TEMPLATE = """
Provide structured, didactic information about the disease: "{disease}".
Return STRICT JSON (no prose outside JSON) with the following schema:

{
  "name": string,
  "summary": string,        // short plain-language overview (2-3 sentences, no advice)
  "statistics": {
    "total_cases": integer,             // a plausible round number
    "incidence_per_100k": number,       // e.g. 123.4
    "recovery_rate": string,            // "xx.x%"
    "mortality_rate": string            // "x.x%"
  },
  "region_breakdown": [                 // 4-8 items max
    {"region": string, "cases": integer, "deaths": integer}
  ],
  "recovery_options": {                 // 3-6 keys
    "<option_name>": "1-3 plain sentences (no medical advice, general info)"
  },
  "medications": [                      // 2-5 items
    {
      "name": string,
      "side_effects": [string, ...],    // 2-5 items
      "dosage": string                  // generic example text
    }
  ],
  "disclaimer": "This content is educational only and not medical advice."
}

Rules:
- Output MUST be valid JSON, UTF-8, with double quotes. No markdown. No extra text.
- Keep numbers realistic but illustrative. If data is uncertain, be conservative.
"""

# ---------------------------
# HELPERS
# ---------------------------
def coerce_percentage(s: str) -> float:
    """Convert '72.5%' -> 72.5 (float)."""
    try:
        return float(str(s).strip().replace('%', '').replace(',', '.'))
    except Exception:
        return 0.0

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Try to load JSON; return {} on failure (and surface an error)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error("Το μοντέλο δεν επέστρεψε έγκυρο JSON.")
        with st.expander("Λεπτομέρειες JSON σφάλματος"):
            st.write(repr(e))
            st.code(text)
        return {}

@st.cache_data(show_spinner=False, ttl=600)
def cached_call_openai(disease: str, model_name: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Call OpenAI with small retries for stability.
    Returns: (ok, data, raw_text_or_error)
    """
    last_error = ""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},  # force JSON
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": USER_TEMPLATE.format(disease=disease)}
                ],
                temperature=0.3,
                timeout=40
            )
            raw = resp.choices[0].message.content
            data = safe_json_loads(raw)
            if data:
                return True, data, raw
            # if JSON invalid, safe_json_loads already showed details
            return False, {}, raw
        except Exception as ex:
            last_error = f"{type(ex).__name__}: {ex}"
            time.sleep(1.2)
    return False, {}, last_error

def render_statistics_block(info: Dict[str, Any]):
    stats = info.get("statistics", {}) or {}
    rec = coerce_percentage(stats.get("recovery_rate", "0%"))
    mort = coerce_percentage(stats.get("mortality_rate", "0%"))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Recovery rate", stats.get("recovery_rate", "—"))
    with c2:
        st.metric("Mortality rate", stats.get("mortality_rate", "—"))
    with c3:
        st.metric("Total cases", f"{stats.get('total_cases', 0):,}")
    with c4:
        st.metric("Incidence / 100k", stats.get("incidence_per_100k", "—"))

    df_rates = pd.DataFrame({"Rate": ["Recovery", "Mortality"], "Value": [rec, mort]}).set_index("Rate")
    st.bar_chart(df_rates)

def render_regions_block(info: Dict[str, Any]):
    regions: List[Dict[str, Any]] = info.get("region_breakdown", []) or []
    if not regions:
        return
    df = pd.DataFrame(regions)
    st.subheader("Regional breakdown")
    st.dataframe(df, use_container_width=True)
    # basic bar chart for cases
    try:
        st.bar_chart(df.set_index("region")["cases"])
    except Exception:
        pass  # don't crash UI if keys missing

def render_recovery_options(info: Dict[str, Any]):
    opts: Dict[str, str] = info.get("recovery_options", {}) or {}
    if not opts:
        return
    st.subheader("Recovery options (general information)")
    for k, v in opts.items():
        st.markdown(f"**{k}**")
        st.write(v)

def render_medications(info: Dict[str, Any]):
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

# ---------------------------
# UI
# ---------------------------
st.title("🩺 Health Insight — OpenAI-only")
st.caption("Εκπαιδευτικό εργαλείο. Δεν παρέχει ιατρικές συμβουλές. Χωρίς εξωτερικά APIs (μόνο OpenAI).")

with st.sidebar:
    st.subheader("Ρυθμίσεις")
    model_name = st.selectbox(
        "Μοντέλο",
        options=["gpt-4o-mini", "gpt-4.1-mini"],
        index=0,
        help="Διάλεξε ελαφρύ και γρήγορο μοντέλο."
    )
    if st.button("🧹 Clear cache"):
        st.cache_data.clear()
        st.success("Cache καθαρίστηκε.")
        st.rerun()

with st.expander("Sanity check (OpenAI)"):
    if st.button("Test OpenAI"):
        try:
            r = client.chat.completions.create(
                model=model_name,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are a JSON echo bot. Only return JSON."},
                    {"role": "user", "content": '{"hello":"world"}'}
                ],
                temperature=0
            )
            st.success("OpenAI OK")
            st.code(r.choices[0].message.content)
        except Exception as e:
            st.error("OpenAI ERROR")
            st.write(repr(e))

st.write("")  # spacing

with st.container():
    disease = st.text_input("Πληκτρολόγησε ασθένεια (π.χ. influenza, diabetes, malaria):", "")
    run = st.button("Ανάλυση")

if run and disease.strip():
    with st.spinner("Φορτώνω…"):
        ok, data, raw = cached_call_openai(disease.strip(), model_name)

    if ok:
        try:
            st.success("ΟΚ — λήψη δεδομένων.")
            st.header(data.get("name", disease.strip()))
            if data.get("summary"):
                st.write(data.get("summary"))
            render_statistics_block(data)
            st.divider()
            col1, col2 = st.columns([1, 1])
            with col1:
                render_regions_block(data)
            with col2:
                render_recovery_options(data)
            st.divider()
            render_medications(data)
            if data.get("disclaimer"):
                st.info(data.get("disclaimer"))

            with st.expander("Raw JSON (debug)"):
                st.code(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception as e:
            st.error("Σφάλμα κατά την εμφάνιση των δεδομένων.")
            with st.expander("Raw / Debug"):
                st.write(repr(e))
                try:
                    st.code(json.dumps(data, ensure_ascii=False, indent=2))
                except Exception:
                    st.write(data)
    else:
        st.error("Αποτυχία κλήσης στο OpenAI. Ξαναδοκίμασε αργότερα.")
        with st.expander("Debug details"):
            # raw may be either the raw JSON text or an error string
            st.write(raw if isinstance(raw, str) else repr(raw))
else:
    st.write("👆 Γράψε μια ασθένεια και πάτα *Ανάλυση* για να ξεκινήσουμε.")


