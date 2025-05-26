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
import openai
import os
from dotenv import load_dotenv
import csv
import io
import requests
import pytz

from openai import OpenAI
client = OpenAI()

# ------------------------
# API Key Setup
# ------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

# ------------------------
# Weather Function
# ------------------------
@st.cache_data(ttl=600)
def get_auckland_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Sunny, 18°C (offline mode)"
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={api_key}"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return "Weather unavailable"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except:
        return "Weather unavailable"

# ------------------------
# Date, Time & Weather Banner
# ------------------------
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

def parse_mixed_date(date_str):
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None

def extract_metadata(text):
    rc_matches = re.findall(r"Application number[:\s]*([\w/-]+)", text, re.IGNORECASE)
    if not rc_matches:
        rc_matches = re.findall(r"RC[0-9]{5,}", text)
    rc_str = "".join(dict.fromkeys(rc_matches))

    company_str = "".join(dict.fromkeys(re.findall(r"Applicant:\s*(.+?)(?=\s*Site address)", text)))
    address_str = "".join(dict.fromkeys(re.findall(r"Site address:\s*(.+?)(?=\s*Legal description)", text)))

    matches_issue = re.findall(r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"Date:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text)
    issue_str = "".join(dict.fromkeys(matches_issue))
    issue_date = parse_mixed_date(issue_str)

    matches_expiry = re.findall(r"shall expire on (\d{1,2} [A-Za-z]+ \d{4})", text) + re.findall(r"expires on (\d{1,2} [A-Za-z]+ \d{4})", text)
    expiry_str = "".join(dict.fromkeys(matches_expiry))
    expiry_date = parse_mixed_date(expiry_str)

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

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer}
    file_exists = os.path.isfile("ai_chat_log.csv")
    with open("ai_chat_log.csv", mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Timestamp", "Question", "Answer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def get_chat_log_as_csv():
    if os.path.exists("ai_chat_log.csv"):
        try:
            df_log = pd.read_csv("ai_chat_log.csv")
            if df_log.empty:
                return None
            output = io.StringIO()
            df_log.to_csv(output, index=False)
            return output.getvalue().encode("utf-8")
        except pd.errors.EmptyDataError:
            return None
    return None

# ------------------------
# Sidebar & Model Loader
# ------------------------
st.sidebar.markdown("""
    <h2 style='color:#2c6e91; font-family:Segoe UI, Roboto, sans-serif;'>
        Control Panel
    </h2>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("Choose LLM model:", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query")

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

# ------------------------
# File Processing & Dashboard
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
        df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))

        # Normalize dates after loading into DataFrame
        df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors='coerce', dayfirst=True)
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True)

        df["Consent Status Enhanced"] = df["Consent Status"]
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"] > datetime.now()) &
            (df["Expiry Date"] <= datetime.now() + timedelta(days=90)),
            "Consent Status Enhanced"
        ] = "Expiring in 90 Days"

        # Metrics
        st.subheader("Consent Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Cons
