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

    rc_str = match(r"Resource Consent Number[:\-]?\s*(.+)")
    company_str = match(r"(Company|Applicant|Organisation) Name[:\-]?\s*(.+)", group=2)
    address_str = match(r"(Location|Site Address|Address)[:\-]?\s*(.+)", group=2)
    triggers_str = match(r"AUP\(OP\) Trigger\(s\)[:\-]?\s*(.+)")
    proposal_str = match(r"Reason for Consent[:\-]?\s*(.+)")
    conditions_numbers = match(r"Consent Condition\(s\)[:\-]?\s*(.+)")
    mitigation_str = match(r"Mitigation[:\-]?\s*(.+)")

    consultant = match(r"(Consultant|Prepared by|Prepared for)[:\-]?\s*(.+)", group=2)
    industry = match(r"(Industry|Sector|Type of Industry)[:\-]?\s*(.+)", group=2)
    location = address_str
    pollutants = match(r"(Pollutants|Emissions|Discharges)[:\-]?\s*(.+)", group=2)

    rules = re.findall(r"E14\.\d+\.\d+", text)
    mitigation = re.findall(r"(bag filter|scrubber|water spray|carbon filter|electrostatic)", text, re.IGNORECASE)

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
        "Filename": filename,
        "Resource Consent Numbers": rc_str,
        "Company Name": company_str,
        "Address": address_str,
        "Consultant": consultant,
        "Industry": industry,
        "Location": location,
        "Pollutants": pollutants,
        "AUP(OP) Triggers": triggers_str,
        "Reason for Consent": proposal_str,
        "Consent Conditions": conditions_numbers,
        "Mitigation (Consent Conditions)": mitigation_str,
        "Mitigation": ", ".join(set(mitigation)) if mitigation else "None Found",
        "Rules Triggered": ", ".join(set(rules)) if rules else "None Found",
        "Consent Date": issue_date_dt,
        "Expiry Date": expiry_date_dt,
        "Expiry Status": status,
        "Duration (years)": duration,
        "Date Validity": invalid_date_note
    }

# Upload and parse PDFs
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)
if not uploaded_files:
    st.warning("Please upload PDF files from a folder to continue.")
    st.stop()

data = []
for file in uploaded_files:
    text = extract_text_from_pdf(file.read())
    metadata = extract_metadata(text, file.name)
    data.append(metadata)

df = pd.DataFrame(data)

# Filters
industries = df['Industry'].dropna().unique().tolist()
statuses = df['Expiry Status'].dropna().unique().tolist()
selected_industries = st.multiselect("Filter by Industry", sorted(industries), default=industries)
selected_statuses = st.multiselect("Filter by Expiry Status", sorted(statuses), default=statuses)
df = df[df['Industry'].isin(selected_industries) & df['Expiry Status'].isin(selected_statuses)]

# Consent summaries
st.markdown("## Consent Document Summaries")
for _, row in df.iterrows():
    with st.expander(f"📄 {row['Filename']} — {row['Company Name']}"):
        st.markdown(f"""
        **Resource Consent Number:** {row['Resource Consent Numbers']}  
        **Company Name:** {row['Company Name']}  
        **Address:** {row['Address']}  
        **Consultant:** {row['Consultant']}  
        **Industry:** {row['Industry']}  
        **Pollutants:** {row['Pollutants']}  
        **Triggers:** {row['AUP(OP) Triggers']}  
        **Reason for Consent:** {row['Reason for Consent']}  
        **Consent Conditions:** {row['Consent Conditions']}  
        **Mitigation (Consent Conditions):** {row['Mitigation (Consent Conditions)']}  
        **Mitigation:** {row['Mitigation']}  
        **Rules Triggered:** {row['Rules Triggered']}  
        **Issue Date:** {row['Consent Date']}  
        **Expiry Date:** {row['Expiry Date']}  
        **Expiry Status:** {row['Expiry Status']}  
        **Duration (years):** {row['Duration (years)']}  
        **Date Validity:** {row['Date Validity']}  
        """)

#########################################

# Summary statistics
st.markdown("<h2 style='color:#144e68;'>Summary Statistics</h2>", unsafe_allow_html=True)
st.metric("Total Consents", len(df))
st.metric("Issued", (df["Expiry Status"] == "Issued").sum())
st.metric("About to Expire", (df["Expiry Status"] == "About to expire").sum())
st.metric("Expired", (df["Expiry Status"] == "Expired").sum())

# Visualization helper
def colored_bar_chart(df, x_col, y_col, title):
    chart = alt.Chart(df).mark_bar(color='#1f77b4').encode(
        x=alt.X(x_col, sort='-y'),
        y=y_col,
        tooltip=[x_col, y_col]
    ).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

# Rule chart
st.subheader("Rule Trigger Frequency")
rule_counts = df["Rules Triggered"].dropna().str.split(", ").explode().value_counts().reset_index()
rule_counts.columns = ["Rule Triggered", "count"]
colored_bar_chart(rule_counts, "Rule Triggered", "count", "AUP E14 Rules Triggered")

# Industry chart
st.subheader("Industry Type Frequency")
industry_counts = df["Industry"].dropna().value_counts().reset_index()
industry_counts.columns = ["Industry", "count"]
colored_bar_chart(industry_counts, "Industry", "count", "Industry Types Involved")

# Pollutant chart
st.subheader("Pollutants Frequency")
pollutant_counts = df["Pollutants"].dropna().str.extractall(r"(PM10|NOx|VOC|SO2|CO2|CO|dust|odour)").value_counts().reset_index()
pollutant_counts.columns = ["Pollutant", "Count"]
colored_bar_chart(pollutant_counts, "Pollutant", "Count", "Pollutant Type Frequency")

# Duration chart
st.subheader("Consent Duration Distribution")
duration_counts = df["Duration (years)"].dropna().value_counts().sort_index().reset_index()
duration_counts.columns = ["Duration (years)", "count"]
colored_bar_chart(duration_counts, "Duration (years)", "count", "Consent Duration in Years")

# Yearly trends
st.subheader("Yearly Trends")
if df["Consent Date"].notna().any():
    df["Consent Year"] = df["Consent Date"].dt.year
    issued_by_year = df["Consent Year"].value_counts().sort_index()
    st.line_chart(issued_by_year.rename("Consents Issued"))

if df["Expiry Date"].notna().any():
    df["Expiry Year"] = df["Expiry Date"].dt.year
    expired_by_year = df[df["Expiry Status"] == "Expired"]["Expiry Year"].value_counts().sort_index()
    st.line_chart(expired_by_year.rename("Consents Expired"))

    about_to_expire_by_year = df[df["Expiry Status"] == "About to expire"]["Expiry Year"].value_counts().sort_index()
    st.line_chart(about_to_expire_by_year.rename("Consents About to Expire"))

# Download filtered files
st.subheader("Download Documents by Type")
doc_type = st.selectbox("Choose document keyword", ["memo", "aee", "aqa", "emp", "aqmp", "dmp", "omp"])
filtered_files = [file for file in uploaded_files if doc_type.lower() in file.name.lower()]

if filtered_files:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for file in filtered_files:
            zipf.writestr(file.name, file.read())
    st.download_button(
        f"Download {doc_type.upper()} Documents", zip_buffer.getvalue(), f"{doc_type}_documents.zip", "application/zip")
else:
    st.info("No matching documents found.")

# Footer
st.markdown("---")
st.caption("Built by Earl Tavera • Alana Jacobson-Pepere • Auckland Air Discharge Intelligence Dashboard • © 2025")
