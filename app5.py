import streamlit as st
import pandas as pd
import os
import fitz  # PyMuPDF
from io import BytesIO
from datetime import datetime, timedelta
import base64
from sentence_transformers import SentenceTransformer, util
import altair as alt
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re

# ------------------- Setup -------------------
st.set_page_config(page_title="Auckland Air Discharge Dashboard", layout="wide")
st.title("Auckland Air Discharge Consents Dashboard")
st.write("Facilitating understanding of industrial air discharges in Auckland (2016 to date)")

@st.cache_resource
def load_bert_model():
    return SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

bert_model = load_bert_model()

# ------------------- Rule Categories -------------------
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

# ------------------- File Upload -------------------
st.sidebar.header("📂 Upload PDF Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDF consent documents",
    type="pdf",
    accept_multiple_files=True
)

if not uploaded_files:
    st.sidebar.warning("📭 Please upload at least one PDF file to continue.")
    st.stop()

pdf_file_paths = uploaded_files

# ------------------- Extractors -------------------
def extract_text_with_fitz(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_with_ocr_fallback(path):
    text = extract_text_with_fitz(path)
    if len(text.strip()) >= 50:
        return text, False
    else:
        try:
            images = convert_from_path(path)
            ocr_text = ""
            for img in images:
                ocr_text += pytesseract.image_to_string(img)
            return ocr_text, True
        except Exception as e:
            raise RuntimeError(f"OCR failed: {e}")

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

# ------------------- Process PDFs -------------------
pdf_texts = []
file_names = []
consent_data = []
failed_files = []
ocr_used_files = []

total_files = len(pdf_file_paths)
st.info(f"📁 Processing {total_files} uploaded PDF file(s)...")
progress_bar = st.progress(0)

for idx, uploaded_file in enumerate(pdf_file_paths):
    file_name = uploaded_file.name
    try:
        # Save uploaded file temporarily
        with open(f"/tmp/{file_name}", "wb") as f:
            f.write(uploaded_file.read())
        temp_path = f"/tmp/{file_name}"

        text, used_ocr = extract_text_with_ocr_fallback(temp_path)
        if used_ocr:
            ocr_used_files.append(file_name)
        metadata = extract_real_metadata(file_name, text)

        file_names.append(file_name)
        pdf_texts.append(text)
        consent_data.append(metadata)

    except Exception as e:
        failed_files.append((file_name, str(e)))

    progress_bar.progress((idx + 1) / total_files)

progress_bar.empty()

# Show summaries
st.success(f"✅ Processed {len(file_names)} / {total_files} file(s).")
if ocr_used_files:
    st.info(f"🔍 OCR applied to {len(ocr_used_files)} scanned file(s): " + ", ".join(ocr_used_files))
if failed_files:
    st.warning(f"⚠️ Skipped {len(failed_files)} file(s):")
    for name, err in failed_files:
        st.text(f" - {name} ➤ {err}")

# ------------------- DataFrame and Filters -------------------
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

# ------------------- Display -------------------
st.subheader("📄 Filtered Consent Table")
st.dataframe(df_filtered)

# ------------------- Dashboard Statistics -------------------
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

# ------------------- Charts -------------------
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

# ------------------- Downloads -------------------
st.subheader("📥 Downloads")
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")

st.download_button("📄 Download Filtered CSV", convert_df_to_csv(df_filtered),
                   file_name="filtered_consents.csv", mime="text/csv")

filtered_filenames = df_filtered["filename"].tolist()
filtered_temp_paths = [f"/tmp/{fname}" for fname in filtered_filenames]

def zip_pdfs(pdf_paths):
    from zipfile import ZipFile
    buffer = BytesIO()
    with ZipFile(buffer, "w") as zipf:
        for path in pdf_paths:
            if os.path.exists(path):
                zipf.write(path, os.path.basename(path))
    return buffer.getvalue()

if filtered_temp_paths:
    zip_bytes = zip_pdfs(filtered_temp_paths)
    b64 = base64.b64encode(zip_bytes).decode()
    href = f'<a href="data:application/zip;base64,{b64}" download="filtered_pdfs.zip">📂 Download PDFs</a>'
    st.markdown(href, unsafe_allow_html=True)

# ------------------- BERT Semantic Search -------------------
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

# ------------------- Footer -------------------
st.caption("© 2025 Auckland Council Dashboard • Earl Tavera & Alana Jacobson‑Pepere")

