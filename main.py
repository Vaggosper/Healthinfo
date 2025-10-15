import streamlit as st
from openai import OpenAI
import json
import pandas as pd
import time
from typing import Any, Dict, List, Tuple
import os


api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)

if not api_key:
    st.error("Λείπει το OpenAI API key. Πρόσθεσέ το στα Secrets ως OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)


SYSTEM_INSTRUCTIONS = """
You are a careful medical information formatter. You NEVER give medical advice.
You ONLY return JSON that fits the schema. If you don't know something, estimate clearly and keep values plausible.
Percentages must be strings with a percent sign (e.g., "72.4%"). Integers must be integers.
"""

# Ζητάμε αυστηρή JSON έξοδο. (χωρίς άλλα APIs, μόνο OpenAI)
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
    """Μετατρέπει '72.5%' -> 72.5"""
    try:
        return float(s.strip().replace('%', '').replace(',', '.'))
    except Exception:
        return 0.0

def safe_json_loads(text: str) -> Dict[str, Any]:
    """Προσπαθεί να φορτώσει JSON, αλλιώς σκάει εξήγηση στο UI."""
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        st.error("Το μοντέλο δεν επέστρεψε έγκυρο JSON. Δοκίμασε ξανά.")
        st.caption(f"JSON decode error: {e}")
        return {}

@st.cache_data(show_spinner=False, ttl=600)
def cached_call_openai(disease: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Κλήση στο OpenAI με μικρό retry για σταθερότητα.
    Επιστρέφει: (ok, data, raw_text)
    """
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                    {"role": "user", "content": USER_TEMPLATE.format(disease=disease)}
                ],
                temperature=0.3,
                timeout=30
            )
            raw = resp.choices[0].message.content
            data = safe_json_loads(raw)
            return (True if data else False, data, raw)
        except Exception as ex:
            if attempt < 2:
                time.sleep(1.2)
            else:
                st.error("Αποτυχία κλήσης στο OpenAI. Ξαναδοκίμασε αργότερα.")
                st.caption(f"{type(ex).__name__}: {ex}")
                return (False, {}, "")
    return (False, {}, "")

def render_statistics_block(info: Dict[str, Any]):
    stats = info.get("statistics", {})
    rec = coerce_percentage(stats.get("recovery_rate", "0%"))
    mort = coerce_percentage(stats.get("mortality_rate", "0%"))

    left, right, _ = st.columns([1,1,1])
    with left:
        st.metric("Recovery rate", stats.get("recovery_rate", "—"))
        st.metric("Mortality rate", stats.get("mortality_rate", "—"))
    with right:
        st.metric("Total cases", f"{stats.get('total_cases', 0):,}")
        st.metric("Incidence / 100k", f"{stats.get('incidence_per_100k', 0)}")


    df_rates = pd.DataFrame(
        {"Rate": ["Recovery", "Mortality"], "Value": [rec, mort]}
    )
    st.bar_chart(df_rates.set_index("Rate"))

def render_regions_block(info: Dict[str, Any]):
    regions: List[Dict[str, Any]] = info.get("region_breakdown", [])
    if not regions:
        return
    df = pd.DataFrame(regions)
    st.subheader("Regional breakdown")
    st.dataframe(df, use_container_width=True)

    st.bar_chart(df.set_index("region")["cases"])

def render_recovery_options(info: Dict[str, Any]):
    opts: Dict[str, str] = info.get("recovery_options", {})
    if not opts:
        return
    st.subheader("Recovery options (general information)")
    for k, v in opts.items():
        st.markdown(f"**{k}**")
        st.write(v)

def render_medications(info: Dict[str, Any]):
    meds: List[Dict[str, Any]] = info.get("medications", [])
    if not meds:
        return
    st.subheader("Medications (examples)")
    for i, m in enumerate(meds, start=1):
        st.markdown(f"**{i}. {m.get('name','—')}**")
        st.write(f"Dosage: {m.get('dosage','—')}")
        se = m.get("side_effects", [])
        if se:
            st.write("Side effects:")
            for s in se:
                st.write(f"· {s}")


st.title("🩺 Health Insight — OpenAI-only")
st.caption("Εκπαιδευτικό εργαλείο. Δεν παρέχει ιατρικές συμβουλές. Χωρίς εξωτερικά APIs (μόνο OpenAI).")

with st.container():
    disease = st.text_input("Πληκτρολόγησε ασθένεια (π.χ. influenza, diabetes, malaria):", "")
    run = st.button("Ανάλυση")

if run and disease.strip():
    with st.spinner("Φορτώνω…"):
        ok, data, raw = cached_call_openai(disease.strip())

    if ok:
        st.success("ΟΚ — λήψη δεδομένων.")
        st.header(data.get("name", disease.strip()))
        if data.get("summary"):
            st.write(data["summary"])
        render_statistics_block(data)
        st.divider()
        col1, col2 = st.columns([1,1])
        with col1:
            render_regions_block(data)
        with col2:
            render_recovery_options(data)
        st.divider()
        render_medications(data)
        if data.get("disclaimer"):
            st.info(data["disclaimer"])
        with st.expander("Raw JSON (debug)"):
            st.code(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        st.error("Κάτι πήγε στραβά με την απόκριση.")
else:
    st.write("👆 Γράψε μια ασθένεια και πάτα *Ανάλυση* για να ξεκινήσουμε.")
