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

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide")

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

    triggers = re.findall(r"E\d+\.\d+\.\d+", text) + re.findall(r"E\d+\.\d+\.", text) + re.findall(r"NES:STO", text) + re.findall(r"NES:AQ", text)
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

@st.cache_resource
def load_bert_model(model_name):
    return SentenceTransformer(model_name)

model_name = st.selectbox("Choose LLM model:", [
    "multi-qa-MiniLM-L6-cos-v1",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])
model = load_bert_model(model_name)

# ------------------------
# Main App Logic
# ------------------------

uploaded_files = st.file_uploader("\U0001F4C2 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)

df = pd.DataFrame()
if uploaded_files:
    all_data = []
    for file in uploaded_files:
        try:
            file_bytes = file.getvalue()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)
            structured_data = extract_metadata(text)
            structured_data["__file_name__"] = file.name
            structured_data["__file_bytes__"] = file_bytes
            all_data.append(structured_data)
        except Exception as e:
            st.error(f"\u274C Error processing {file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        df["GeoKey"] = df["Address"].str.lower().str.strip()
        latitudes, longitudes = [], []

        for addr in df["GeoKey"]:
            lat, lon = geocode_address(addr)
            latitudes.append(lat)
            longitudes.append(lon)

        df["Latitude"] = latitudes
        df["Longitude"] = longitudes

        total_consents = len(df)
        expired_consents = df["Consent Status"].value_counts().get("Expired", 0)
        active_consents = df["Consent Status"].value_counts().get("Active", 0)

        # About to expire in 90 days
        today = datetime.now()
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce')
        about_to_expire = df[(df["Expiry Date"].notnull()) & (df["Expiry Date"] > today) & (df["Expiry Date"] <= today + timedelta(days=90))]
        about_to_expire_count = len(about_to_expire)

        st.markdown(f"<h4 style='color:#228B22;'><b>Processed {total_consents} PDF file(s)</b></h4>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Consents Uploaded", total_consents)
        col2.metric("Total Expired Consents", expired_consents)
        col3.metric("Expiring in 90 Days", about_to_expire_count)

        st.markdown("<h4><b>Consent Summary Table</b></h4>", unsafe_allow_html=True)
        st.dataframe(df.drop(columns=["Text Blob", "__file_bytes__", "__file_name__", "GeoKey"]))
        
        csv = df.drop(columns=["Text Blob", "__file_bytes__", "__file_name__", "GeoKey"]).to_csv(index=False).encode("utf-8")
        st.download_button("\U0001F4E5 Download CSV", data=csv, file_name="consent_summary.csv", mime="text/csv")
        
        map_df = df.dropna(subset=["Latitude", "Longitude"])
        if not map_df.empty:
            st.markdown("<h4><b>Consent Locations Map (Mapbox)</b></h4>", unsafe_allow_html=True)
            fig = px.scatter_mapbox(
                map_df,
                lat="Latitude",
                lon="Longitude",
                hover_name="Company Name",
                hover_data={"Address": True, "Consent Status": True},
                zoom=10,
                height=500
            )
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_traces(marker=dict(size=12, color="blue"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid geolocation data available to display a map.")

        chart_df = pd.DataFrame({
            "Consent Status": ["Expired", "Active"],
            "Count": [expired_consents, active_consents]
        })
        bar_fig = px.bar(chart_df, x="Consent Status", y="Count", title="Expired vs Active Consents", text="Count")
        bar_fig.update_traces(marker_color=["crimson", "green"], textposition="outside")
        st.plotly_chart(bar_fig)

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
                st.markdown(f"**{i+1}. {row['Company Name']} — {row['Address']}**")
                st.markdown(f"- Triggers: `{row['AUP(OP) Triggers']}`")
                st.markdown(f"- Reason: {row['Reason for Consent']}")
                st.markdown(f"- Status: `{row['Consent Status']}` | Expires: `{row['Expiry Date']}`")
                st.download_button(
                    label="📄 Download Original PDF",
                    data=row.get("__file_bytes", b""),
                    file_name=row.get("__file_name", f"document_{i+1}.pdf"),
                    mime="application/pdf",
                    key=f"semantic_download_{i}"
                )
                st.markdown("---")
else:
    st.info("\U0001F4C4 Please upload one or more PDF files to begin.")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence Dashboard © 2025")
