
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
    doc = fitz.open(stream=file_path_or_stream, filetype="pdf") if isinstance(file_path_or_stream, BytesIO) else fitz.open(file_path_or_stream)
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

pdf_texts = []
file_names = []
consent_data = []
failed_files = []
ocr_used_files = []

if upload_mode == "📂 Folder Path (Local)":
    folder_path = st.sidebar.text_input("Enter the path to folder containing PDFs").strip().strip('"').strip("'").rstrip("/")
    recursive = st.sidebar.checkbox("Include subfolders", value=True)
    if not os.path.isdir(folder_path):
        st.sidebar.warning("📭 Please enter a valid folder path.")
        st.stop()

    def list_pdf_files(folder_path, recursive=True):
        pdf_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
            if not recursive:
                break
        return sorted(pdf_files)

    pdf_paths = list_pdf_files(folder_path, recursive)
    if not pdf_paths:
        st.warning("⚠️ No PDF files found in the folder.")
        st.stop()

    st.info(f"📁 Processing {len(pdf_paths)} file(s)...")
    progress_bar = st.progress(0)

    for idx, path in enumerate(pdf_paths):
        file_name = os.path.basename(path)
        try:
            text, used_ocr = extract_text_with_ocr_fallback(path)
            if used_ocr:
                ocr_used_files.append(file_name)
            metadata = extract_real_metadata(file_name, text)
            file_names.append(file_name)
            pdf_texts.append(text)
            consent_data.append(metadata)
        except Exception as e:
            failed_files.append((file_name, str(e)))
        progress_bar.progress((idx + 1) / len(pdf_paths))

    progress_bar.empty()

else:
    uploaded_files = st.sidebar.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    if not uploaded_files:
        st.warning("📭 Please upload one or more PDF files to begin.")
        st.stop()

    st.info(f"📄 Processing {len(uploaded_files)} uploaded file(s)...")
    progress_bar = st.progress(0)

    for idx, uploaded_file in enumerate(uploaded_files):
        file_name = uploaded_file.name
        try:
            with BytesIO(uploaded_file.read()) as f:
                text, used_ocr = extract_text_with_ocr_fallback(f)
            if used_ocr:
                ocr_used_files.append(file_name)
            metadata = extract_real_metadata(file_name, text)
            file_names.append(file_name)
            pdf_texts.append(text)
            consent_data.append(metadata)
        except Exception as e:
            failed_files.append((file_name, str(e)))
        progress_bar.progress((idx + 1) / len(uploaded_files))

    progress_bar.empty()

st.success(f"✅ Processed {len(file_names)} file(s).")
if ocr_used_files:
    st.info("🔍 OCR fallback applied to: " + ", ".join(ocr_used_files))
if failed_files:
    st.warning("⚠️ Skipped file(s):")
    for name, err in failed_files:
        st.text(f" - {name} ➤ {err}")

df = pd.DataFrame(consent_data)
st.sidebar.header("🔍 Filter Options")
selected_industries = st.sidebar.multiselect("Industry", df["industry"].unique())
selected_pollutants = st.sidebar.multiselect("Pollutants", df["pollutants"].unique())
selected_location = st.sidebar.multiselect("Suburb", df["location"].unique())

df_filtered = df.copy()
if selected_industries:
    df_filtered = df_filtered[df_filtered["industry"].isin(selected_industries)]
if selected_pollutants:
    df_filtered = df_filtered[df_filtered["pollutants"].str.contains("|".join(selected_pollutants))]
if selected_location:
    df_filtered = df_filtered[df_filtered["location"].isin(selected_location)]

st.subheader("📄 Filtered Consent Table")
st.dataframe(df_filtered)

st.subheader("📊 Dashboard Statistics")
today = datetime.today()
df_filtered["consent_date"] = pd.to_datetime(df_filtered["consent_date"], errors="coerce")
df_filtered["expiry_date"] = pd.to_datetime(df_filtered["expiry_date"], errors="coerce")

total_issued = df_filtered.shape[0]
expired = df_filtered[df_filtered["expiry_date"] < today].shape[0]
about_to_expire = df_filtered[
    (df_filtered["expiry_date"] >= today) &
    (df_filtered["expiry_date"] <= today + timedelta(days=180))
].shape[0]

col1, col2, col3 = st.columns(3)
col1.metric("📝 Total Consents Issued", total_issued)
col2.metric("⏳ Expiring Soon", about_to_expire)
col3.metric("❌ Expired", expired)

st.subheader("📈 Visual Summaries")
col4, col5 = st.columns(2)
with col4:
    st.markdown("**Consultants**")
    st.bar_chart(df_filtered["consultant"].value_counts())
with col5:
    st.markdown("**Industries**")
    st.bar_chart(df_filtered["industry"].value_counts())

col6, col7 = st.columns(2)
with col6:
    st.markdown("**Pollutants**")
    st.bar_chart(df_filtered["pollutants"].value_counts())
with col7:
    st.markdown("**Suburbs**")
    st.bar_chart(df_filtered["location"].value_counts())

df_filtered["rule_category"] = df_filtered["rules_triggered"].apply(map_rule_category)
st.subheader("🧾 AUP E14 Rule Categories")
rule_counts = df_filtered["rule_category"].value_counts().reset_index()
rule_counts.columns = ["Category", "Count"]

chart = alt.Chart(rule_counts).mark_arc().encode(
    theta=alt.Theta(field="Count", type="quantitative"),
    color=alt.Color(field="Category", type="nominal"),
    tooltip=["Category", "Count"]
).properties(title="Rule Category Distribution")

st.altair_chart(chart, use_container_width=True)

st.subheader("📥 Downloads")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button("📄 Download Filtered CSV", convert_df_to_csv(df_filtered),
                   file_name="filtered_consents.csv", mime="text/csv")

st.subheader("Semantic Search (BERT)")
query = st.text_input("🔍 Ask a question (e.g. mitigation for dust in South Auckland)")

if query:
    with st.spinner("Searching..."):
        query_embedding = bert_model.encode(query, convert_to_tensor=True)
        doc_embeddings = bert_model.encode(pdf_texts, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, doc_embeddings)[0]
        top_k = min(5, len(pdf_texts))
        top_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    st.markdown("### 🔎 Top Matches")
    for idx, score in top_results:
        st.markdown(f"**📄 File:** `{file_names[idx]}` — **Score:** `{score.item():.2f}`")
        with st.expander("📖 View Excerpt"):
            st.write(pdf_texts[idx][:1500] + "...")

st.caption("© 2025 Auckland Council Dashboard • Earl Tavera & Alana Jacobson‑Pepere")
