# Auckland Air Discharge Consent Dashboard - Complete with Gemini Fallback

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
import os
from dotenv import load_dotenv
import csv
import io
import requests
import pytz
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit config
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

# Weather
@st.cache_data(ttl=600)
def get_auckland_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Sunny, 18°C (offline mode)"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={api_key}"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return "Weather unavailable"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except:
        return "Weather unavailable"

nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
today = nz_time.strftime("%A, %d %B %Y")
current_time = nz_time.strftime("%I:%M %p")
weather = get_auckland_weather()

st.markdown(f"""
<div style='text-align:center; padding:12px; font-size:1.2em; background-color:#656e6b;
            border-radius:10px; margin-bottom:15px; font-weight:500; color:white;'>
    📅 <strong>{today}</strong> &nbsp;&nbsp;&nbsp; ⏰ <strong>{current_time}</strong> &nbsp;&nbsp;&nbsp; 🌦️ <strong>{weather}</strong> &nbsp;&nbsp;&nbsp; 📍 <strong>Auckland</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='color:#2c6e91; text-align:center; font-size:2.7em; font-family: Quicksand, sans-serif;'>
    Auckland Air Discharge Consent Dashboard
</h1>
""", unsafe_allow_html=True)

# -----------------------
# Utility Functions
# -----------------------

def parse_mixed_date(date_str):
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except:
            continue
    return None

def check_expiry(expiry_date):
    if expiry_date is None:
        return "Unknown"
    return "Expired" if expiry_date < datetime.now() else "Active"

def extract_metadata(text):
    rc = re.findall(r"Application number[:\s]*([\w/-]+)", text, re.IGNORECASE) or re.findall(r"RC[0-9]{5,}", text)
    rc_str = "".join(dict.fromkeys(rc))
    company = "".join(dict.fromkeys(re.findall(r"Applicant:\s*(.+?)(?=\s*Site address)", text)))
    address = "".join(dict.fromkeys(re.findall(r"Site address:\s*(.+?)(?=\s*Legal description)", text)))

    issue_match = re.findall(r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"Date:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text)
    expiry_match = re.findall(r"shall expire on (\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"expires on (\d{1,2} [A-Za-z]+ \d{4})", text)

    issue_date = parse_mixed_date("".join(issue_match))
    expiry_date = parse_mixed_date("".join(expiry_match))

    triggers = " ".join(dict.fromkeys(re.findall(r"E\\d+\\.\\d+\\.\\d+|E\\d+\\.\\d+|NES:STO|NES:AQ", text)))
    proposal = " ".join(re.findall(r"Proposal\\s*:\\s*(.+?)(?=\\n[A-Z]|\\.)", text, re.DOTALL))
    conditions_str = "".join(re.findall(r"(?<=Conditions).*?(?=Advice notes)", text, re.DOTALL))
    conditions = ", ".join(re.findall(r"^\\d+(?=\\.)", conditions_str, re.MULTILINE))
    mitigation = ", ".join(dict.fromkeys([f"{w} Management Plan" for w in re.findall(r"(?i)\\b(\\w+)\\sManagement Plan", conditions_str)]))

    return {
        "Resource Consent Numbers": rc_str,
        "Company Name": company,
        "Address": address,
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown",
        "AUP(OP) Triggers": triggers,
        "Reason for Consent": proposal,
        "Consent Conditions": conditions,
        "Mitigation (Consent Conditions)": mitigation,
        "Consent Status": check_expiry(expiry_date),
        "Text Blob": text
    }

@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

# -----------------------
# File Upload & Processing
# -----------------------

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query")
model_name = st.sidebar.selectbox("Choose LLM model:", [
    "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"
])
@st.cache_resource
def load_model(name): return SentenceTransformer(name)
model = load_model(model_name)

if uploaded_files:
    all_data = []
    for file in uploaded_files:
        file_bytes = file.read()
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        data = extract_metadata(text)
        data["__file_name__"] = file.name
        data["__file_bytes__"] = file_bytes
        all_data.append(data)

    df = pd.DataFrame(all_data)
    df["GeoKey"] = df["Address"].str.lower().str.strip()
    df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))
    df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors='coerce', dayfirst=True)
    df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True)

    df["Consent Status Enhanced"] = df["Consent Status"]
    df.loc[
        (df["Consent Status"] == "Active") &
        (df["Expiry Date"] > datetime.now()) &
        (df["Expiry Date"] <= datetime.now() + timedelta(days=90)),
        "Consent Status Enhanced"
    ] = "Expiring in 90 Days"

    st.subheader("Consent Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Consents", len(df))
    col2.metric("Expired", df["Consent Status"].value_counts().get("Expired", 0))
    col3.metric("Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum())

    # Status chart
    status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
    status_counts.columns = ["Consent Status", "Count"]
    fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status",
                        color_discrete_map={"Unknown":"gray", "Expired":"red", "Active":"green", "Expiring in 90 Days":"orange"})
    fig_status.update_layout(title="Consent Status Overview", title_x=0.5)
    st.plotly_chart(fig_status, use_container_width=True)

    with st.expander("Consent Table", expanded=True):
        status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist())
        filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
        display_df = filtered_df[[
            "__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date",
            "Consent Status Enhanced", "AUP(OP) Triggers", "Reason for Consent", "Mitigation (Consent Conditions)"
        ]].rename(columns={"__file_name__": "File Name", "Consent Status Enhanced": "Consent Status"})
        st.dataframe(display_df)
        st.download_button("Download CSV", display_df.to_csv(index=False).encode("utf-8"), "filtered_consents.csv")

    with st.expander("Consent Map", expanded=True):
        map_df = df.dropna(subset=["Latitude", "Longitude"])
        if not map_df.empty:
            fig = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", hover_name="Company Name",
                hover_data={"Address": True, "Consent Status Enhanced": True, "Issue Date": True, "Expiry Date": True},
                zoom=10, height=500, color="Consent Status Enhanced",
                color_discrete_map={"Unknown":"gray", "Expired":"red", "Active":"green", "Expiring in 90 Days":"orange"})
            fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Ask AI About Consents Chatbot (Gemini)
# ----------------------------
st.markdown("### 🤖 Ask AI About Consents")
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)
    chat_input = st.text_area("Search any query:", key="chat_input")
    if st.button("Ask AI"):
        if not chat_input.strip():
            st.warning("Please enter any query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    context_sample = df[[
                        "Company Name", "Consent Status", "AUP(OP) Triggers", 
                        "Mitigation (Consent Conditions)", "Expiry Date"
                    ]].dropna().head(10).to_dict(orient="records")

                    gemini_prompt = f"""
You are an assistant that helps analyze environmental resource consents.
Use the following sample data from industrial air discharge consents to answer the user query.

---
Sample Data:
{context_sample}

---
Query: {chat_input}

Answer with bullet points and highlight key terms where needed.
"""
                    response = genai.generate_text(model="gemini-pro", prompt=gemini_prompt)
                    answer_raw = response.result
                    st.markdown(f"### 🧠 Answer from Gemini AI\n\n{answer_raw}")
                except Exception as e:
                    st.error(f"AI error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True
)
