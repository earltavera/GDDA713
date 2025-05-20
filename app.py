# --------------------------------------------
# Auckland Air Discharge Consent Dashboard
# --------------------------------------------

import streamlit as st
import pandas as pd
import pymupdf
fitz = pymupdf
import re
from datetime import datetime, timedelta
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import base64

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

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
# Utility Functions
# ------------------------

def check_expiry(expiry_date):
    if expiry_date is None:
        return "Unknown"
    return "Expired" if expiry_date < datetime.now() else "Active"

@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

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
# Sidebar and Upload Controls
# ------------------------
st.sidebar.title("Control Panel")
model_name = st.sidebar.selectbox("Choose LLM model:", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query")

# ------------------------
# Load Model
# ------------------------
@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

# ------------------------
# File Processing & Main Dashboard
# ------------------------
if uploaded_files:
    all_data = []
    for file in uploaded_files:
        try:
            file_bytes = file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            data = extract_metadata(text)
            data["__file_name__"] = file.name
            data["__file_bytes__"] = file_bytes
            all_data.append(data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df["GeoKey"] = df["Address"].str.lower().str.strip()
        lat, lon = [], []
        for address in df["GeoKey"]:
            latitude, longitude = geocode_address(address)
            lat.append(latitude)
            lon.append(longitude)
        df["Latitude"] = lat
        df["Longitude"] = lon

        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True)

        st.subheader("Consent Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"<h3 style='color:#1f77b4'>{len(df)} Total Consents</h3>", unsafe_allow_html=True)
        col2.markdown(f"<h3 style='color:#d62728'>{df['Consent Status'].value_counts().get('Expired', 0)} Expired</h3>", unsafe_allow_html=True)
        exp_soon = df[(df["Expiry Date"] > datetime.now()) & (df["Expiry Date"] <= datetime.now() + timedelta(days=90))]
        col3.markdown(f"<h3 style='color:#ff9900'>{len(exp_soon)} Expiring in 90 Days</h3>", unsafe_allow_html=True)

        with st.expander("Consent Table", expanded=True):
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status"].unique().tolist())
            filtered_df = df if status_filter == "All" else df[df["Consent Status"] == status_filter]
            st.dataframe(filtered_df.rename(columns={"__file_name__": "File Name"})[["File Name"] + [col for col in filtered_df.columns if col not in ["__file_name__", "Text Blob", "__file_bytes__", "GeoKey"]]])

            csv = filtered_df.rename(columns={"__file_name__": "File Name"})[["File Name"] + [col for col in filtered_df.columns if col not in ["__file_name__", "Text Blob", "__file_bytes__", "GeoKey"]]].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "consents_summary.csv", "text/csv")

        with st.expander("Consent Map", expanded=True):
            map_df = df.dropna(subset=["Latitude", "Longitude"])
            if not map_df.empty:
                fig = px.scatter_mapbox(
    map_df,
    lat="Latitude",
    lon="Longitude",
    hover_name="Company Name",
    hover_data={
        "Address": True,
        "Consent Status": True,
        "Issue Date": True,
        "Expiry Date": True,
        "AUP(OP) Triggers": True,
        "Mitigation (Consent Conditions)": True
    },
    zoom=10,
    height=500
)
                fig.update_layout(mapbox_style="open-street-map")
                st.plotly_chart(fig, use_container_width=True)

        with st.expander("Consent Status Chart", expanded=True):
            chart_df = df["Consent Status"].value_counts().reset_index()
            chart_df.columns = ["Status", "Count"]
            fig = px.bar(chart_df, x="Status", y="Count", text="Count", color="Status")
            st.plotly_chart(fig)

        with st.expander("Semantic Search Results", expanded=True):
            if query_input:
                corpus = df["Text Blob"].tolist()
                corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
                query_embedding = model.encode(query_input, convert_to_tensor=True)
                scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_k = scores.argsort(descending=True)[:3]
                for i, idx in enumerate(top_k):
                    row = df.iloc[idx.item()]
                    st.markdown(f"**{i+1}. {row['Company Name']} - {row['Address']}**")
                    st.markdown(f"- **Triggers**: `{row['AUP(OP) Triggers']}`")
                    st.markdown(f"- **Expires**: `{row['Expiry Date'].strftime('%d-%m-%Y') if pd.notnull(row['Expiry Date']) else 'Unknown'}`")
                    st.download_button(
                        f"📄 Download PDF ({row['__file_name__']})",
                        data=row['__file_bytes__'],
                        file_name=row['__file_name__'],
                        mime="application/pdf",
                        key=f"download_{i}"
                    )
                    st.markdown("---")
else:
    st.info("📄 Please upload one or more PDF files to begin.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence Dashboard © 2025")
