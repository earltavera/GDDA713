# Auckland Air Discharge Consent Dashboard
import streamlit as st
import pandas as pd
import pymupdf
fitz = pymupdf
import zipfile
from io import BytesIO
from datetime import datetime
import re
import altair as alt
from PIL import Image
import pytesseract

# Set Streamlit page config
st.set_page_config(page_title="Auckland Air Discharge Dashboard", layout="wide")
st.markdown("<h1 style='color:#2c6e91;'>Auckland Industrial Air Discharge Consent Dashboard</h1>", unsafe_allow_html=True)

# PDF Text Extraction
def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    if len(text.strip()) < 50:
        try:
            text = ""
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img)
        except Exception as e:
            text += f"\n[OCR failed: {e}]"
    return text

# Metadata Extraction
def extract_metadata(text, filename):
    def match(pattern, group=1, default="Not Found"):
        result = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return result.group(group).strip().replace('\n', ' ') if result else default

    rc_str = match(r"(Resource Consent Number|Consent No\.?|Application Number|RCN|Consent #:?)[:\-]?\s*(.+)", group=2)
    company_str = match(r"(Company Name|Applicant Name|Organisation Name|Company|Applicant|Organisation)[:\-]?\s*(.+)", group=2)
    address_str = match(r"(Location|Site Address|Address)[:\-]?\s*(.+)", group=2)
    triggers_str = match(r"(AUP\(OP\)|AUP)[\s\-:]*Trigger[s]*[:\-]?\s*(.{3,100})", group=2)
    proposal_str = match(r"(Reason for Consent|Proposal|Purpose)[:\-]?\s*(.{3,200})", group=2)
    conditions_numbers = match(r"(Consent Condition[s]*|Conditions Applied)[:\-]?\s*(.{3,200})", group=2)
    mitigation_str = match(r"(Mitigation Measures|Mitigation)[:\-]?\s*(.{3,200})", group=2)

    expiry_date_str = match(r"Expiry Date[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", group=1, default=None)
    issue_date_str = match(r"(Consent Date|Application Date|Applied Date|Date of Consent)[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})", group=2, default=None)

    try:
        expiry_date_dt = pd.to_datetime(expiry_date_str, dayfirst=True, errors='coerce')
        issue_date_dt = pd.to_datetime(issue_date_str, dayfirst=True, errors='coerce')

        if pd.isna(issue_date_dt) or pd.isna(expiry_date_dt):
            possible_dates = re.findall(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}", text)
            parsed_dates = pd.to_datetime(possible_dates, dayfirst=True, errors='coerce').dropna()
            if pd.isna(issue_date_dt) and not parsed_dates.empty:
                issue_date_dt = parsed_dates.min()
            if pd.isna(expiry_date_dt) and len(parsed_dates) > 1:
                expiry_date_dt = parsed_dates.max()

        duration = (expiry_date_dt.year - issue_date_dt.year) if not pd.isna(expiry_date_dt) and not pd.isna(issue_date_dt) else None
    except:
        expiry_date_dt = None
        issue_date_dt = None
        duration = None

    now = pd.Timestamp.now()
    if pd.notna(expiry_date_dt):
        if expiry_date_dt < now:
            status = "Expired"
        elif expiry_date_dt < now + pd.DateOffset(months=6):
            status = "About to expire"
        else:
            status = "Issued"
    else:
        status = "Unknown"

    invalid_date_note = "Invalid or missing" if pd.isna(issue_date_dt) or pd.isna(expiry_date_dt) else "Valid"

    return {
        "Resource Consent Number": rc_str,
        "Company Name": company_str,
        "Address": address_str,
        "Issue Date": issue_date_dt,
        "Expiry Date": expiry_date_dt,
        "Expiry Status": status,
        "Duration (years)": duration,
        "Date Validity": invalid_date_note,
        "AUP(OP) Triggers": triggers_str,
        "Reason for Consent": proposal_str,
        "Consent Conditions": conditions_numbers,
        "Mitigation": mitigation_str
    }

# Upload PDF files
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.warning("Please upload PDF files from a folder to continue.")
    st.stop()

# Process each file
data = []
for file in uploaded_files:
    text = extract_text_from_pdf(file.read())
    metadata = extract_metadata(text, file.name)
    data.append(metadata)

# Create dataframe
df = pd.DataFrame(data)

# ===========================
# FILTERS FOR EXPIRY STATUS & RANGE
# ===========================
if 'Expiry Date' in df.columns:
    df['Years to Expiry'] = ((df['Expiry Date'] - pd.Timestamp.now()).dt.days / 365).round(1)

    # Filter by expiry status
    expiry_status_options = df['Expiry Status'].dropna().unique().tolist()
    selected_statuses = st.multiselect("Filter by Expiry Status", expiry_status_options, default=expiry_status_options)

    # Filter by years to expiry
    min_years = int(df['Years to Expiry'].min(skipna=True)) if not df['Years to Expiry'].isna().all() else -10
    max_years = int(df['Years to Expiry'].max(skipna=True)) if not df['Years to Expiry'].isna().all() else 10
    selected_year_range = st.slider("Filter by Years to Expiry", min_years, max_years, (min_years, max_years))

    df = df[df['Expiry Status'].isin(selected_statuses) &
            df['Years to Expiry'].between(selected_year_range[0], selected_year_range[1], inclusive='both')]

# ===========================
# EXPIRY ANALYSIS CHART
# ===========================
if 'Expiry Date' in df.columns:
    st.subheader("Expiry Status Breakdown by Years to Expiry")
    df['Years to Expiry'] = ((df['Expiry Date'] - pd.Timestamp.now()).dt.days / 365).round(1)
    expiry_bins = pd.cut(df['Years to Expiry'], bins=[-100, -1, 0, 1, 3, 5, 10, 50], right=False)
    expiry_chart = expiry_bins.value_counts().sort_index().reset_index()
    expiry_chart.columns = ["Years Range", "Count"]
    st.bar_chart(data=expiry_chart.set_index("Years Range"))
