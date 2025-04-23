import streamlit as st
import pandas as pd
import os
import fitz  # PyMuPDF
from io import BytesIO
from datetime import datetime, timedelta
import base64
from sentence_transformers import SentenceTransformer, util
import altair as alt
from PIL import Image
import re

st.set_page_config(page_title="Auckland Air Discharge Dashboard", layout="wide")
st.title("Auckland Air Discharge Consents Dashboard")
st.write("Facilitating understanding of industrial air discharges in Auckland (2016 to date)")

@st.cache_resource
def load_bert_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

bert_model = load_bert_model()

IS_CLOUD = os.getenv("IS_STREAMLIT_CLOUD") is not None

RULE_CATEGORIES = {
    "E14.6.1": "Controlled",
    "E14.6.2": "Restricted Discretionary",
    "E14.6.3": "Discretionary",
    "E14.6.4": "Non-Complying",
    "E14.6.5": "Permitted"
}

def map_rule_category(rule_code):
    for rule in str(rule_code).split(","):
        rule = rule.strip()
        if rule in RULE_CATEGORIES:
            return RULE_CATEGORIES[rule]
    return "Unclassified"

def extract_text_with_fitz(file_path_or_stream):
    if isinstance(file_path_or_stream, BytesIO):
        doc = fitz.open(stream=file_path_or_stream, filetype="pdf")
    elif isinstance(file_path_or_stream, str) and os.path.exists(file_path_or_stream):
        doc = fitz.open(file_path_or_stream)
    else:
        raise ValueError("Unsupported input type for extract_text_with_fitz")
    return "\n".join([page.get_text() for page in doc])

def extract_text_with_ocr_fallback(file_like):
    try:
        text = extract_text_with_fitz(file_like)
        if len(text.strip()) >= 50:
            return text, False
        return "[OCR skipped — not available in cloud mode]", True
    except Exception as e:
        return f"[Failed to extract text: {e}]", True

def extract_real_metadata(file_name, text):
    def find_match(pattern, default="Unknown", flags=re.IGNORECASE):
        match = re.search(pattern, text, flags)
        return match.group(1).strip() if match else default

    industry = find_match(r"Industry[:\-]?\s*(.*)")
    location = find_match(r"Location[:\-]?\s*(.*)")
    consultant = find_match(r"Consultant[:\-]?\s*(.*)")
    consent_date = find_match(r"(?:Consent|Application|Applied) Date[:\-]?\s*(\d{4}-\d{2}-\d{2})", default=None)
    expiry_date = find_match(r"Expiry Date[:\-]?\s*(\d{4}-\d{2}-\d{2})", default=None)

    try:
        if consent_date and expiry_date:
            duration_years = (pd.to_datetime(expiry_date) - pd.to_datetime(consent_date)).days // 365
        else:
            duration_years = None
    except:
        duration_years = None

    pollutants = ", ".join(sorted(set(re.findall(r"(PM10|NOx|SO2|CO2|VOC|dust|odour)", text, flags=re.IGNORECASE))))
    rules = ", ".join(sorted(set(re.findall(r"(E14\.\d+\.\d+)", text))))
    mitigation = find_match(r"Mitigation Measures?[:\-]?\s*(.*)")

    return {
        "filename": file_name,
        "industry": industry,
        "pollutants": pollutants,
        "location": location,
        "consent_date": consent_date,
        "expiry_date": expiry_date,
        "duration_years": duration_years,
        "rules_triggered": rules,
        "mitigation": mitigation,
        "consultant": consultant,
    }

st.sidebar.header("📁 Upload Options")
upload_mode = st.sidebar.radio("Choose how to provide PDFs:", ["📂 Folder Path (Local)", "📄 Upload Files"])

if IS_CLOUD and upload_mode == "📂 Folder Path (Local)":
    st.sidebar.error("⚠️ Folder access is not supported on Streamlit Cloud. Use 'Upload Files' mode instead.")
    st.stop()

# Everything else stays the same...
# To keep this clean, the remaining logic (PDF processing, display, charts, etc.)
# can be appended to this updated starter block

# Saving this updated version
