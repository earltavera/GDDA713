# --------------------------------------------
# Auckland Air Discharge Consent Dashboard
# --------------------------------------------

import streamlit as st
import pandas as pd
import pymupdf
fitz = pymupdf
import re
from datetime import datetime
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import st_folium

# Set Streamlit layout and styling
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide")

# ------------------------
# Custom Styling
# ------------------------
st.markdown("""
    <style>
    h1 {
        color: #2c6e91;
        text-align: center;
        font-size: 2.5em;
    }
    .metric-label {
        font-weight: bold !important;
        color: #003366;
    }
    .stDataFrame {
        background-color: #ffffff !important;
    }
    .stPlotlyChart {
        background-color: #f9f9ff !important;
        padding: 1rem;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>Auckland Air Discharge Consent Dashboard</h1>", unsafe_allow_html=True)

# ------------------------
# Expiry Check
# ------------------------
def check_expiry(expiry_date):
    if expiry_date is None:
        return "Unknown"
    return "Expired" if expiry_date < datetime.now() else "Active"

# ------------------------
# Metadata Extraction
# ------------------------
def extract_metadata(text):
    rc_str = "".join(dict.fromkeys(re.findall(r"Application number:\s*(.+?)(?=\s*Applicant)", text)))
    company_str = "".join(dict.fromkeys(re.findall(r"Applicant:\s*(.+?)(?=\s*Site address)", text)))
    address_str = "".join(dict.fromkeys(re.findall(r"Site address:\s*(.+?)(?=\s*Legal description)", text)))

    matches_issue = re.findall(r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})", text)
    issue_str = "".join(dict.fromkeys(matches_issue))
    try:
        issue_date = datetime.strptime(issue_str, "%d %B %Y")
    except:
        issue_date = None

    matches_expiry = re.findall(r"shall expire on (\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"expires on (\d{1,2} [A-Za-z]+ \d{4})", text)
    expiry_str = "".join(dict.fromkeys(matches_expiry))
    try:
        expiry_date = datetime.strptime(expiry_str, "%d %B %Y")
    except:
        expiry_date = None

    triggers = re.findall(r"E\d+\.\d+\.\d+", text) + re.findall(r"E\d+\.\d+.", text) + re.findall(r"NES:STO", text) + re.findall(r"NES:AQ", text)
    triggers_str = " ".join(dict.fromkeys(triggers))

    proposal_str = " ".join(re.findall(r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\.)", text, re.DOTALL))

    conditions_str = "".join(re.findall(r"(?<=Conditions).*?(?=Advice notes)", text, re.DOTALL))
    conditions_numbers = re.findall(r"^\d+(?=\.)", conditions_str, re.MULTILINE)
    managementplan_final = list(dict.fromkeys([f"{word} Management Plan" for word in re.findall(r"(?i)\b(\w+)\sManagement Plan", conditions_str)]))

    return {
        "Resource Consent Numbers": rc_str,
        "Company Name": company_str,
        "Address": address_str,
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown",
        "AUP(OP) Triggers": triggers_str,
        "Reason for Consent": proposal_str,
        "Consent Conditions": ", ".join(conditions_numbers),
        "Mitigation (Consent Conditions)": ", ".join(managementplan_final),
        "Consent Status": check_expiry(expiry_date),
        "Text Blob": text
    }

# ------------------------
# BERT model for semantic search
# ------------------------
@st.cache_resource
def load_bert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_bert_model()

def highlight_keywords(text, query):
    for word in query.lower().split():
        text = re.sub(f"(?i)({word})", r"<mark>\\1</mark>", text)
    return text

# ------------------------
# File Upload and Processing
# ------------------------
uploaded_files = st.file_uploader("\U0001F4C2 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

df = pd.DataFrame()
if uploaded_files:
    all_data = []
    for file in uploaded_files:
        try:
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            structured_data = extract_metadata(text)
            all_data.append(structured_data)
        except Exception as e:
            st.error(f"\u274C Error processing {file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        total_consents = len(df)
        expired_consents = df["Consent Status"].value_counts().get("Expired", 0)
        active_consents = df["Consent Status"].value_counts().get("Active", 0)

        st.markdown(f"<h4 style='color:#228B22;'><b>Processed {total_consents} PDF file(s)</b></h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Total Consents Uploaded", total_consents)
        col2.metric("Total Expired Consents", expired_consents)

        # Filters
        st.markdown("<h4><b>Filter Consents</b></h4>", unsafe_allow_html=True)
        selected_status = st.multiselect("Filter by Consent Status", options=df["Consent Status"].unique(), default=df["Consent Status"].unique())
        selected_company = st.multiselect("Filter by Company Name", options=df["Company Name"].unique(), default=df["Company Name"].unique())
        filtered_df = df[df["Consent Status"].isin(selected_status) & df["Company Name"].isin(selected_company)]

        st.markdown("<h4><b>Consent Summary Table</b></h4>", unsafe_allow_html=True)
        st.dataframe(filtered_df.drop(columns=["Text Blob"]))

        chart_df = pd.DataFrame({
            "Consent Status": ["Expired", "Active"],
            "Count": [expired_consents, active_consents]
        })
        bar_fig = px.bar(chart_df, x="Consent Status", y="Count", title="Expired vs Active Consents", text="Count")
        bar_fig.update_traces(marker_color=["crimson", "green"], textposition="outside")
        st.plotly_chart(bar_fig)

        pie_fig = px.pie(chart_df, names="Consent Status", values="Count", title="Consent Status Distribution")
        pie_fig.update_traces(textinfo='label+percent', pull=[0.05, 0])
        st.plotly_chart(pie_fig)

        # Map with Geopy and Folium
        st.markdown("<h4><b>Map of Consent Locations</b></h4>", unsafe_allow_html=True)
        geolocator = Nominatim(user_agent="air_discharge_dashboard")
        df["Coordinates"] = df["Address"].apply(lambda x: geolocator.geocode(x) if x else None)
        df["Lat"] = df["Coordinates"].apply(lambda loc: loc.latitude if loc else None)
        df["Lon"] = df["Coordinates"].apply(lambda loc: loc.longitude if loc else None)

        valid_coords_df = df.dropna(subset=["Lat", "Lon"])
        map_ = folium.Map(location=[-36.8485, 174.7633], zoom_start=10)

        for _, row in valid_coords_df.iterrows():
            folium.Marker(
                location=[row["Lat"], row["Lon"]],
                popup=f"{row['Company Name']}<br>{row['Address']}",
                tooltip=row["Company Name"]
            ).add_to(map_)

        st_folium(map_, width=700, height=450)

        # CSV download
        csv = df.drop(columns=["Text Blob"]).to_csv(index=False).encode("utf-8")
        st.download_button("\U0001F4E5 Download CSV", data=csv, file_name="consent_summary.csv", mime="text/csv")

        # Semantic Search
        st.markdown("<h4><b>Semantic Search</b></h4>", unsafe_allow_html=True)
        query = st.text_input("Ask a question (e.g., 'expired consents in Onehunga', 'dust mitigation')")

        if query and not df.empty:
            st.markdown("**Top 3 matching consents:**")
            corpus = df["Text Blob"].tolist()
            corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
            query_embedding = model.encode(query, convert_to_tensor=True)

            scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
            top_k_idx = scores.argsort(descending=True)[:3]

            for i, idx in enumerate(top_k_idx):
                row = df.iloc[idx.item()]
                snippet = row["Text Blob"][:600]
                highlighted = highlight_keywords(snippet, query)
                st.markdown(f"**{i+1}. {row['Company Name']} — {row['Address']}**")
                st.markdown(f"- Triggers: `{row['AUP(OP) Triggers']}`")
                st.markdown(f"- Reason: {row['Reason for Consent']}")
                st.markdown(f"- Status: `{row['Consent Status']}` | Expires: `{row['Expiry Date']}`")
                st.markdown(f"<div style='background-color:#f1f1f1;padding:8px;border-radius:8px'>{highlighted}...</div>", unsafe_allow_html=True)
                st.markdown("---")
else:
    st.info("\U0001F4C4 Please upload one or more PDF files to begin.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence Dashboard © 2025")
