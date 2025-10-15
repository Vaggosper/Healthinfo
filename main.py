# main.py
import os
import re
import time
import json
from typing import Any, Dict, List, Tuple, Optional

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

MODEL_NAME = "gpt-4o-mini"  # γρήγορο/φθηνό (ensure available in your account)

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

def ensure_pct_str(s: Any) -> str:
    if s is None:
        return "0%"
    t = str(s).strip()
    if not t:
        return "0%"
    # If it's numeric, append %
    try:
        float(t.replace("%", "").replace(",", "."))
        if not t.endswith("%"):
            return t + "%"
        return t
    except Exception:
        # not numeric, return as-is
        return t

def extract_json_block(text: str) -> str:
    """Πάρε το πρώτο {...} block ακόμη κι αν έχει κείμενο γύρω-γύρω."""
    if not isinstance(text, str):
        return ""
    # Attempt to find the most plausible JSON object by matching braces
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else ""

def safe_load_json(text: Any) -> Dict[str, Any]:
    """
    Try to load a JSON string, or if it's already a dict-like object, return it.
    """
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return {}
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

def sanitize_info(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coerce some fields to the expected types to avoid rendering errors.
    This is lightweight and conservative.
    """
    if not isinstance(info, dict):
        return {}

    stats = info.get("statistics", {}) or {}
    # total_cases -> int
    try:
        stats["total_cases"] = int(stats.get("total_cases", 0) or 0)
    except Exception:
        # try to parse strings like "1,234"
        try:
            stats["total_cases"] = int(str(stats.get("total_cases", 0)).replace(",", "").split(".")[0])
        except Exception:
            stats["total_cases"] = 0

    # incidence_per_100k -> float
    try:
        stats["incidence_per_100k"] = float(str(stats.get("incidence_per_100k", 0)).replace(",", "."))
    except Exception:
        stats["incidence_per_100k"] = 0.0

    # recovery_rate, mortality_rate -> percent strings
    stats["recovery_rate"] = ensure_pct_str(stats.get("recovery_rate", "0%"))
    stats["mortality_rate"] = ensure_pct_str(stats.get("mortality_rate", "0%"))

    info["statistics"] = stats

    # region_breakdown: ensure integers for cases/deaths
    rlist = info.get("region_breakdown", []) or []
    fixed_regions = []
    if isinstance(rlist, list):
        for r in rlist:
            if not isinstance(r, dict):
                continue
            try:
                cases = int(r.get("cases", 0) or 0)
            except Exception:
                try:
                    cases = int(str(r.get("cases", 0)).replace(",", ""))
                except Exception:
                    cases = 0
            try:
                deaths = int(r.get("deaths", 0) or 0)
            except Exception:
                try:
                    deaths = int(str(r.get("deaths", 0)).replace(",", ""))
                except Exception:
                    deaths = 0
            fixed_regions.append({"region": str(r.get("region", "") or ""), "cases": cases, "deaths": deaths})
    info["region_breakdown"] = fixed_regions

    # medications: ensure list of objects with expected keys
    meds = info.get("medications", []) or []
    fixed_meds = []
    if isinstance(meds, list):
        for m in meds:
            if not isinstance(m, dict):
                continue
            fixed_meds.append({
                "name": str(m.get("name", "") or ""),
                "dosage": str(m.get("dosage", "") or ""),
                "side_effects": [str(s) for s in (m.get("side_effects", []) or []) if s]
            })
    info["medications"] = fixed_meds

    # recovery_options: ensure dict of string->string
    ropts = info.get("recovery_options", {}) or {}
    if isinstance(ropts, dict):
        info["recovery_options"] = {str(k): str(v) for k, v in ropts.items()}
    else:
        info["recovery_options"] = {}

    return info

@st.cache_data(show_spinner=False, ttl=600)
def call_openai(disease: str) -> Tuple[bool, Dict[str, Any], str]:
    """
    Use Responses API with a single input string.
    Returns (ok, data, raw_or_error).
    """
    user_text = USER_TEMPLATE.format(disease=disease)
    prompt = SYSTEM_INSTRUCTIONS.strip() + "\n\n" + user_text.strip()

    last = ""
    # Simple exponential backoff
    for attempt in range(1, 4):
        try:
            resp = client.responses.create(
                model=MODEL_NAME,
                response_format={"type": "json_object"},
                temperature=0.2,
                max_output_tokens=1200,
                input=prompt,  # string input for Responses API
            )

            # Best-effort extraction of structured JSON from various SDK shapes:
            parsed: Optional[Dict[str, Any]] = None
            raw_text: str = ""

            # 1) If SDK returned a parsed object in the top-level (sometimes happens)
            if hasattr(resp, "output") and getattr(resp, "output") is not None:
                try:
                    # resp.output may be a list with items containing content entries
                    # Try to find a content entry with a 'type' that implies JSON or a direct dict value
                    out_list = getattr(resp, "output")
                    # If it's already a dict-like, try to find JSON-like content
                    for item in out_list:
                        # item.content might be a list of dicts
                        content = getattr(item, "content", None) or item.get("content", []) if isinstance(item, dict) else None
                        if isinstance(content, list):
                            for c in content:
                                # JSON payload
                                if isinstance(c, dict) and ("json" in (c.get("type", "") or "") or c.get("type") == "json"):
                                    val = c.get("value") or c.get("data") or c.get("json") or c.get("value")
                                    if isinstance(val, dict):
                                        parsed = val
                                        break
                                # direct value
                                if isinstance(c, dict) and isinstance(c.get("value", None), dict):
                                    parsed = c.get("value")
                                    break
                                # text
                                if isinstance(c, dict) and isinstance(c.get("text", None), str):
                                    raw_text += c.get("text", "")
                            if parsed:
                                break
                    # fallback: some SDKs have resp.output_text convenience attribute
                except Exception:
                    pass

            # 2) Resp might contain convenience attribute output_text
            if not parsed:
                raw_text = getattr(resp, "output_text", "") or raw_text

            # 3) If no raw_text yet, try to compose from resp.output text parts
            if not raw_text and hasattr(resp, "output") and getattr(resp, "output") is not None:
                try:
                    out_list = getattr(resp, "output")
                    for item in out_list:
                        content = getattr(item, "content", None) or (item.get("content", []) if isinstance(item, dict) else [])
                        if isinstance(content, list):
                            for c in content:
                                if isinstance(c, dict):
                                    raw_text += str(c.get("text", "") or "")
                except Exception:
                    pass

            # 4) If the SDK returned a top-level 'response' or similar dict with parsed data
            if not parsed:
                # try generic attributes that might contain parsed JSON
                for attr in ("output_parsed", "parsed", "response", "data"):
                    v = getattr(resp, attr, None)
                    if isinstance(v, dict):
                        parsed = v
                        break
                    # some SDKs use resp.get(...)
                    if isinstance(resp, dict) and resp.get(attr) and isinstance(resp.get(attr), dict):
                        parsed = resp.get(attr)
                        break

            # If we have parsed JSON as a dict, use that. Otherwise attempt to json.loads raw_text
            if parsed and isinstance(parsed, dict):
                data = parsed
                raw = json.dumps(parsed, ensure_ascii=False, indent=2)
                return True, sanitize_info(data), raw
            else:
                # try to read raw_text as JSON string
                data = safe_load_json(raw_text)
                if data:
                    return True, sanitize_info(data), raw_text
                # final fallback: if resp itself is string-like
                if isinstance(resp, str):
                    data = safe_load_json(resp)
                    if data:
                        return True, sanitize_info(data), resp

            # If we reached here, parsing failed
            last = "Invalid JSON from model"
        except Exception as ex:
            last = f"{type(ex).__name__}: {ex}"
        # Exponential backoff
        time.sleep(0.5 * (2 ** (attempt - 1)))
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


