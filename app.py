# Auckland Air Discharge Consent Dashboard
import streamlit as st
import pandas as pd
import pymupdf
fitz = pymupdf  # Ensure correct fitz from PyMuPDF
import zipfile
from io import BytesIO
from datetime import datetime
import re
import altair as alt

st.set_page_config(page_title="Auckland Air Discharge Dashboard", layout="wide")
st.markdown("<h1 style='color:#2c6e91;'>Auckland Industrial Air Discharge Consent Dashboard</h1>", unsafe_allow_html=True)

# Upload PDF files
uploaded_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.warning("Please upload PDF files from a folder to continue.")
    st.stop()

# Helper function to extract text from PDF using fitz (PyMuPDF)
def extract_text_from_pdf(file_bytes):
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)
    if len(text.strip()) < 50:
        try:
            from PIL import Image
            import pytesseract
            text = ""
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text += pytesseract.image_to_string(img)
        except Exception as e:
            text += f"\n[OCR failed: {e}]"
    return text

# Helper to extract metadata from text
def extract_metadata(text, filename):
    def match(pattern, group=1, default=None):
        result = re.search(pattern, text, re.IGNORECASE)
        return result.group(group).strip() if result else default

    rules = re.findall(r"E14\.\d+\.\d+", text)
    mitigation = re.findall(r"(bag filter|scrubber|water spray|carbon filter|electrostatic)", text, re.IGNORECASE)
    expiry_date = match(r"Expiry Date[:\-]?\s*(\d{4}-\d{2}-\d{2})")
    issue_date = match(r"(Consent|Application|Applied) Date[:\-]?\s*(\d{4}-\d{2}-\d{2})", 2)

    try:
        expiry_date_dt = pd.to_datetime(expiry_date)
        issue_date_dt = pd.to_datetime(issue_date)
        duration = expiry_date_dt.year - issue_date_dt.year
    except:
        expiry_date_dt = None
        issue_date_dt = None
        duration = None

    now = pd.Timestamp.now()
    if expiry_date_dt:
        if expiry_date_dt < now:
            status = "Expired"
        elif expiry_date_dt < now + pd.DateOffset(months=6):
            status = "About to expire"
        else:
            status = "Issued"
    else:
        status = "Unknown"

    return {
        "Filename": filename,
        "Consultant": match(r"Consultant[:\-]?\s*(.+)"),
        "Industry": match(r"Industry[:\-]?\s*(.+)"),
        "Location": match(r"Location[:\-]?\s*(.+)"),
        "Pollutants": match(r"Pollutants[:\-]?\s*(.+)"),
        "Mitigation": ", ".join(set(mitigation)) if mitigation else None,
        "Rules Triggered": ", ".join(set(rules)) if rules else None,
        "Consent Date": issue_date_dt,
        "Expiry Date": expiry_date_dt,
        "Expiry Status": status,
        "Duration (years)": duration
    }

# Process each PDF and extract metadata
data = []
for file in uploaded_files:
    text = extract_text_from_pdf(file.read())
    metadata = extract_metadata(text, file.name)
    data.append(metadata)

# Create DataFrame and CSV download
df = pd.DataFrame(data)
st.success(f"Extracted data from {len(df)} files.")

# Filter preview
st.markdown(f"<h3 style='color:#1f77b4;'>{"Filter and Preview Data".strip('"')}</h3>", unsafe_allow_html=True)
doc_keyword = st.text_input("Filter filename by keyword (e.g. memo, AEE, EMP):").lower()
filtered_df = df[df["Filename"].str.lower().str.contains(doc_keyword)] if doc_keyword else df
st.dataframe(filtered_df)

csv = df.to_csv(index=False)
st.download_button("Download CSV", csv, "air_discharge_consents.csv", "text/csv")

# Summary statistics
st.markdown(f"<h2 style='color:#144e68;'>{"Summary Statistics".strip('"')}</h2>", unsafe_allow_html=True)
st.metric("Total Consents", len(df))
st.metric("Issued", (df["Expiry Status"] == "Issued").sum())
st.metric("About to Expire", (df["Expiry Status"] == "About to expire").sum())
st.metric("Expired", (df["Expiry Status"] == "Expired").sum())

# Bar chart helper using Altair with color
def colored_bar_chart(df, x_col, y_col, title):
    chart = alt.Chart(df).mark_bar(color='#1f77b4').encode(
        x=alt.X(x_col, sort='-y'),
        y=y_col,
        tooltip=[x_col, y_col]
    ).properties(title=title)
    st.altair_chart(chart, use_container_width=True)

st.subheader("Rule Trigger Frequency")
rule_counts = df["Rules Triggered"].dropna().str.split(", ").explode().value_counts().reset_index()
rule_counts.columns = ["Rule Triggered", "count"]
colored_bar_chart(rule_counts, "Rule Triggered", "count", "AUP E14 Rules Triggered")

st.subheader("Industry Type Frequency")
industry_counts = df["Industry"].dropna().value_counts().reset_index()
industry_counts.columns = ["Industry", "count"]
colored_bar_chart(industry_counts, "Industry", "count", "Industry Types Involved")

st.subheader("Pollutants Frequency")
pollutant_counts = df["Pollutants"].dropna().str.extractall(r"(PM10|NOx|VOC|SO2|CO2|CO|dust|odour)").value_counts().reset_index()
pollutant_counts.columns = ["Pollutant", "Count"]
colored_bar_chart(pollutant_counts, "Pollutant", "Count", "Pollutant Type Frequency")

st.subheader("Consent Duration Distribution")
duration_counts = df["Duration (years)"].dropna().value_counts().sort_index().reset_index()
duration_counts.columns = ["Duration (years)", "count"]
colored_bar_chart(duration_counts, "Duration (years)", "count", "Consent Duration in Years")

# Yearly Trends
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

# Download by document type
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
