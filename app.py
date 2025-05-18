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

# Highlight rows with invalid/missing dates
highlighted_df = df.style.apply(
    lambda x: ["background-color: #ffcccc" if x["Date Validity"] == "Invalid or missing" else "" for _ in x],
    axis=1
)

st.success(f"Extracted data from {len(df)} files.")

# Filter preview
st.markdown("<h3 style='color:#1f77b4;'>Filter and Preview Data</h3>", unsafe_allow_html=True)
doc_keyword = st.text_input("Filter filename by keyword (e.g. memo, AEE, EMP):").lower()
filtered_df = df[df["Filename"].str.lower().str.contains(doc_keyword)] if doc_keyword else df

# Reapply highlighting to filtered preview
highlighted_filtered_df = filtered_df.style.apply(
    lambda x: ["background-color: #ffcccc" if x["Date Validity"] == "Invalid or missing" else "" for _ in x],
    axis=1
)
st.dataframe(highlighted_filtered_df, use_container_width=True)

# CSV download
csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "air_discharge_consents.csv", "text/csv")

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
