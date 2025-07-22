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
import os
from dotenv import load_dotenv
import csv
import io
import requests
import pytz # Import pytz for timezone handling
import json
import time # ADDED for progress bar UX

# --- LLM Specific Imports ---
import google.generativeai as genai # For Gemini
from langchain_groq import ChatGroq # For Groq (Langchain integration)
from langchain_core.messages import SystemMessage, HumanMessage # Needed for Langchain messages
# --- End LLM Specific Imports ---

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
# OpenWeatherMap API key
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿", initial_sidebar_state="expanded")

if google_api_key:
    genai.configure(api_key=google_api_key)

# --- Initialize Session State ---
if 'master_df' not in st.session_state:
    st.session_state.master_df = pd.DataFrame()
if 'corpus_embeddings' not in st.session_state:
    st.session_state.corpus_embeddings = None
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""
# Add a key for the file_uploader to session state
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0
# Initialize the state variable to control the semantic search input
if 'semantic_search_query' not in st.session_state:
    st.session_state.semantic_search_query = ""


# --- Weather Function ---
@st.cache_data(ttl=600)
def get_auckland_weather():
    if not openweathermap_api_key:
        return "Sunny, 18°C (offline mode)" # Default / fallback
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={openweathermap_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data.get("cod") != 200:
            return "Weather unavailable"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except requests.exceptions.RequestException:
        return "Weather unavailable (network error)"
    except Exception:
        return "Weather unavailable (data error)"

# --- Date, Time & Weather Banner ---
nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
today = nz_time.strftime("%A, %d %B %Y")
current_time = nz_time.strftime("%I:%M %p")
weather = get_auckland_weather()

st.markdown(f"""
    <div style='text-align:center; padding:12px; font-size:1.2em; background-color:#656e6b;
                    border-radius:10px; margin-bottom:15px; font-weight:500; color:white;'>
        📍 <strong>Auckland</strong> &nbsp;&nbsp;&nbsp; 📅 <strong>{today}</strong> &nbsp;&nbsp;&nbsp; ⏰ <strong>{current_time}</strong> &nbsp;&nbsp;&nbsp; 🌦️ <strong>{weather}</strong>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center;">
        <h2 style='color:#004489; font-family: Quicksand, sans-serif; font-size: 2.7em;'>
            Auckland Air Discharge Consent Dashboard
        </h2>
        <p style='font-size: 1.1em; color: #dc002e;'>
            This dashboard allows you to upload Air Discharge Resource Consent Decision Reports to transform your files into meaningful data.
            Explore the data using the CSV file options, or interact with the data using Gemini AI, Groq AI, or LLM Semantic Query.
        </p>
    </div>
    <br>
""", unsafe_allow_html=True)

st.markdown("---")
with st.expander("About the Auckland Air Discharge Consent Dashboard", expanded=False):
    st.write("""
    Kia Ora! Welcome to the **Auckland Air Discharge Consent Dashboard**, a pioneering tool designed to revolutionize how we interact with critical environmental data. In Auckland, managing **Air Discharge Resource Consents** is vital for maintaining our air quality and ensuring regulatory compliance. Traditionally, this information has been locked away in numerous, disparate PDF reports, making it incredibly challenging to access, analyze, and monitor effectively.

    This dashboard addresses that very challenge head-on. We've developed a user-friendly, web-based application that automatically extracts, visualizes, and analyzes data from these PDF consent reports. Our key innovation lies in leveraging **Artificial Intelligence (AI)**, including **Large Language Models (LLMs)**, to transform static documents into dynamic, searchable insights. This means you can now effortlessly track consent statuses, identify expiring permits, and even query the data using natural language, asking questions like, "Which companies have expired consents?" or "What conditions apply to dust emissions?".

    Ultimately, this dashboard is more than just a data viewer; it's a strategic asset for proactive environmental management. By providing immediate access to comprehensive, intelligent insights, it empowers regulators, businesses, and stakeholders to ensure ongoing compliance, make informed decisions, and contribute to a healthier, more sustainable Auckland.
    """)
st.markdown("---")

# --- Utility Functions ---
def localize_to_auckland(dt):
    if pd.isna(dt) or not isinstance(dt, datetime):
        return pd.NaT
    auckland_tz = pytz.timezone("Pacific/Auckland")
    if dt.tzinfo is None:
        return auckland_tz.localize(dt, is_dst=None)
    else:
        return dt.astimezone(auckland_tz)

def check_expiry(expiry_date):
    if pd.isna(expiry_date):
        return "Unknown"
    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
    localized_expiry_date = expiry_date
    if expiry_date.tzinfo is None:
        localized_expiry_date = localize_to_auckland(expiry_date)
    return "Expired" if localized_expiry_date < current_nz_time else "Active"

@st.cache_data(show_spinner=False)
def geocode_address(address):
    if not address or pd.isna(address):
        return (None, None)
    standardized_address = address.strip()
    if not re.search(r'auckland', standardized_address, re.IGNORECASE):
        standardized_address += ", Auckland"
    if not re.search(r'new zealand|nz', standardized_address, re.IGNORECASE):
        standardized_address += ", New Zealand"
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    try:
        location = geocode(standardized_address)
        if location:
            return (location.latitude, location.longitude)
    except Exception:
        pass
    return (None, None)

def extract_metadata(text):
    # RC number patterns
    rc_patterns = [
        r"Application number:\s*(.+?)(?=\s*Applicant:)",
        r"Application numbers:\s*(.+?)(?=\s*Applicant:)",
        r"Application number(s):\s*(.+?)(?=\s*Applicant:)",
        r"RC[0-9]{5,}"
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE))
    flattened_rc_matches = []
    for item in rc_matches:
        if isinstance(item, tuple):
            flattened_rc_matches.append(item[-1])
        else:
            flattened_rc_matches.append(item)
    rc_str = ", ".join(list(dict.fromkeys(flattened_rc_matches)))

    # Company name patterns
    company_patterns = [
        r"Applicant:\s*(.+?)(?=\s*Site address)",
        r"Applicant's name:\s*(.+?)(?=\s*Site address)"
    ]
    company_matches = []
    for pattern in company_patterns:
        company_matches.extend(re.findall(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE))
    company_str = ", ".join(list(dict.fromkeys(company_matches)))

    # Address patterns
    address_patterns = [
        r"Site address:\s*(.+?)(?=\s*Legal description)",
        r"Site address:\s*(.+?)(?=\s*Activity status)",
        r"Site address:\s*(.+?)(?=\s*Proposal)",
        r"Site address:\s*(.+?)(?=\n\s*\n)"
    ]
    address_str = ""
    for pattern in address_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            address_str = match.group(1).strip().replace('\n', ', ')
            break

    # Issue date patterns
    issue_date_patterns = [
        r"Commissioner\s*(\d{1,2} [A-Za-z]+ \d{4})", r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})", r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)", r"(\b\d{2}/\d{2}/\d{2}\b)"
    ]
    issue_date = None
    for pattern in issue_date_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            for dt_str_candidate in matches:
                dt_str = dt_str_candidate[0] if isinstance(dt_str_candidate, tuple) and dt_str_candidate else dt_str_candidate
                if not isinstance(dt_str, str) or not dt_str.strip():
                    continue
                try:
                    if '/' in dt_str:
                        issue_date = datetime.strptime(dt_str, "%d/%m/%y") if len(dt_str.split('/')[-1]) == 2 else datetime.strptime(dt_str, "%d/%m/%Y")
                    else:
                        dt_str = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', r'\1', dt_str)
                        issue_date = datetime.strptime(dt_str, "%d %B %Y")
                    break
                except ValueError:
                    continue
            if issue_date:
                break

    # Consent Expiry patterns
    expiry_patterns = [
        r"expire\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", r"expires\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expires\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})", r"expire\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})", r"expires\s+([A-Za-z]+\s+years)",
        r"expire\s+([A-Za-z]+\s+years)", r"DIS\d{5,}(?:-w+)?\b\s+will\s+expire\s+(\d{1,}\s+years)",
        r"expires\s+(\d{1,}\s+months\s+[A-Za-z])+\s+[.?!]", r"expires\s+on\s+(\d{1,2}(?:st|nd|rd|th)?\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expires\s+on\s+the\s+(\d{1,2}(?:st|nd|rd|th)?\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)", r"expire\s+on\s+(\d{1,2}/\d{1,2}/\d{4})",
        r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})", r"expire\s+([A-Za-z]+\s+(\d{1,})\s+years)",
        r"expire\s+(\d{1,2}\s+years)", r"expires\s+(\d{1,2}\s+years)",
        r"expire\s+([A-Za-z]+\s+(\d{1,2})\s+[A-Za-z]+)", r"earlier\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"on\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)", r"on\s+the\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"(\d{1,}\s+years)",
    ]
    expiry_date = None
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            for dt_val_candidate in matches:
                dt_str = dt_val_candidate[0] if isinstance(dt_val_candidate, tuple) and dt_val_candidate else dt_val_candidate
                if not isinstance(dt_str, str) or not dt_str.strip():
                    continue
                try:
                    if '/' in dt_str:
                        expiry_date = datetime.strptime(dt_str, "%d/%m/%y") if len(dt_str.split('/')[-1]) == 2 else datetime.strptime(dt_str, "%d/%m/%Y")
                    else:
                        dt_str_cleaned = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?(?: of)?\b', r'\1', dt_str)
                        expiry_date = datetime.strptime(dt_str_cleaned, "%d %B %Y")
                    break
                except ValueError:
                    continue
            if expiry_date:
                break
    
    # If no expiry date is found, try to infer it from phrases like "shall expire 15 years from the date of issue"
    if not expiry_date:
        # Match patterns like "shall expire 15 years from the date of issue"
        match = re.search(
            r"(expire[s]?|shall expire)[^\n]{0,40}?(\d{1,2})\s+years\s+from\s+the\s+date\s+of\s+issue", text,
            re.IGNORECASE
        )
        if match and issue_date:
            try:
                years = int(match.group(2))
                expiry_date = issue_date + timedelta(days=years * 365.25)  # leap-year adjusted
            except Exception:
                expiry_date = None

    # Fallback: match generic "X years" if the above fails
    if not expiry_date:
        years_match = re.search(r'(\d{1,2})\s+years', text, re.IGNORECASE)
        if years_match and issue_date:
            try:
                num_years = int(years_match.group(1))
                expiry_date = issue_date + timedelta(days=num_years * 365.25)
            except Exception:
                expiry_date = None

    # Final expiry string for display (ensure this is set before return)
    expiry_str = expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown Expiry Date"

    # AUP(OP) Triggers pattern
    # This pattern now specifically finds "E14.4.1" and then captures the content within the following parentheses.
    trigger_pattern = r'E14\.4\.1\s*\(([^)]+)\)'
    triggers = re.findall(trigger_pattern, text, re.IGNORECASE)
    triggers_str = ", ".join(list(dict.fromkeys(triggers)))


    # Proposal patterns
    proposal_patterns= [
        r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\.)", r"Proposal\s*(.+?)(?=\n[A-Z]|\.)",
        r"Proposal\s*(.+?)(?=\n[A-Z]|\:)", r"Introduction and summary of proposal\s*(.+?)\s*Submissions",
        r"Proposal, site and locality description\s*(.+?)(?=\n[A-Z]|\.)", r"Summary of Decision\s*(.+?)(?=\n[A-Z]|\.)",
        r"Summary of proposal and activity status\s*(.+?)(?=\n[A-Z]|\.)"
    ]
    proposal = [match for pattern in proposal_patterns for match in re.findall(pattern, text, re.MULTILINE | re.DOTALL)]
    proposal_str = "".join(list(dict.fromkeys(proposal)))

    # Conditions patterns
    conditions_patterns = [
        r"(?:Specific conditions - Air Discharge DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions -)", r"(?:Air Quality conditions).*?(?=Wastewater Discharge conditions)",
        r"(?:Air Discharge Permit Conditions).*?(?=E\. Definitions)", r"(?:Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=DIS\d{5,}(?:-\w+)?\b)",
        r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)", r"(?:Conditions Specific to air quality).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:regional discharge DIS\d{5,}(?:-w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?:Specific conditions - Air Discharge DIS\d{5,}\b).*?(?=Advice notes)", r"(?:Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=Advice notes)", r"(?:DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=DIS\d{5,}(?:-\w+)?\b - Specific conditions)",
        r"(?:Specific Conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)", r"(?:Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b Only).*?(?=Advice notes)",
        r"(?:Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Specific Conditions -)", r"(?:SPECIFIC CONDITIONS - DISCHARGE TO AIR DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Conditions relevant to Discharge Permit DIS\d{5,}(?:-\w+)?\b only).*?(?=Advice notes)", r"(?:Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit (DIS\d{5,}(?:-\w+)?\b)).*?(?=Advice notes)", r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b (air)).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specifc conditions)", r"(?:Attachment 1: Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?:Specific Air Discharge Conditions).*?(?=Advice notes)", r"(?:Specific conditions - Discharge to Air: DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - discharge permit (air discharge) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:Air Discharge Limits).*?(?= Acoustic Conditions)",
        r"(?:Specific conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:Specific conditions - air discharge permit (s15) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Secific conditions)", r"(?:Specific conditions relating to Air discharge permit - DIS\d{5,}(?:-\w+)?\b).*?(?=General Advice notes)",
        r"(?:Specific conditions - Discharge permit (s15) - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?:Specific Conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?:Specific conditions - Discharge to air: DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)", r"(?:Attachement 1: Consolidated conditions of consent as amended).*?(?=Resource Consent Notice of Works Starting)",
        r"(?:Specific conditions - Air Discharge consent - DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)", r"(?:Specific conditions - Discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:DIS\d{5,}(?:-\w+)?\b - Air Discharge).*?(?=SUB\d{5,}\b) - Subdivision", r"(?:DIS\d{5,}(?:-\w+)?\b & DIS\d{5,}(?:-\w+)?\b).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?:Specific conditions - Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice Notes - General)", r"(?:AIR QUALITY - ROCK CRUSHER).*?(?=GROUNDWATER)",
        r"(?<=Conditions).*?(?=Advice notes)"
    ]
    conditions_str = ""
    for pattern in conditions_patterns:
        conditions_match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if conditions_match:
            conditions_str = conditions_match.group(0).strip()
            break

    return {
        "Resource Consent Numbers": rc_str or "Unknown",
        "Company Name": company_str or "Unknown",
        "Address": address_str or "Unknown",
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_str,
        "AUP(OP) Triggers": triggers_str or "None Found",
        "Reason for Consent": proposal_str or "Unknown",
        "Consent Conditions": conditions_str or "Unknown",
        "Consent Status": check_expiry(expiry_date),
        "Text Blob": text
    }

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer):
    timestamp = datetime.now(pytz.timezone("Pacific/Auckland")).strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer}
    file_exists = os.path.isfile("ai_chat_log.csv")
    try:
        with open("ai_chat_log.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Timestamp", "Question", "Answer"])
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)
    except Exception as e:
        st.error(f"Error logging chat history: {e}")

def get_chat_log_as_csv():
    if os.path.exists("ai_chat_log.csv"):
        try:
            df_log = pd.read_csv("ai_chat_log.csv")
            return None if df_log.empty else df_log.to_csv(index=False).encode("utf-8")
        except pd.errors.EmptyDataError:
            return None
    return None

def set_chat_input(query):
    st.session_state.chat_input = query

# --- Sidebar & Model Loader ---
st.sidebar.markdown("<h2 style='color:#1E90FF; font-family:Segoe UI, Roboto, sans-serif;'>Control Panel</h2>", unsafe_allow_html=True)
model_name = st.sidebar.selectbox("Choose Embedding Model:", ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"])

# Updated file_uploader with dynamic key
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can add more files later without losing your current data.",
    key=f"pdf_uploader_{st.session_state.file_uploader_key}" # Dynamic key
)
# Modified query_input to use the new semantic_search_query state variable
query_input = st.sidebar.text_input(
    "LLM Semantic Search Query",
    value=st.session_state.semantic_search_query, # Control its value
    key="query_input_unique_key" # Give it a unique key to avoid conflict with `value` param
)


if st.sidebar.button("Clear All Data", type="primary"):
    st.session_state.master_df = pd.DataFrame()
    st.session_state.corpus_embeddings = None
    st.session_state.semantic_search_query = ""
    st.session_state.chat_input = ""
    st.session_state.file_uploader_key += 1
    st.rerun()

# --- "About Us" Section in Sidebar ---
st.sidebar.markdown("---")
with st.sidebar.expander("About the Creators", expanded=False):
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alana_image_path = os.path.join(script_dir, "assets", "Alana.jpg")
    earl_image_path = os.path.join(script_dir, "assets", "Earl_images.jpg")

    col1_sidebar, col2_sidebar = st.columns(2)
    with col1_sidebar:
        if os.path.exists(alana_image_path):
            st.image(alana_image_path, caption="Alana Jacobson-Pepere", use_container_width=True)
        else:
            st.warning("Image file 'Alana.jpg' not found.")

    with col2_sidebar:
        if os.path.exists(earl_image_path):
            st.image(earl_image_path, caption="Earl Tavera", use_container_width=True)
        else:
            st.warning("Image file 'Earl_images.jpg' not found.")

    st.write("""
    This dashboard was developed by **Alana Jacobson-Pepere** and **Earl Tavera**, Data Analytics Students at NZSE GDDA7224C.

    They are passionate about leveraging technology to simplify complex data, empower informed decision-making, and contribute to the sustainable management of our city's resources.
    """)
st.sidebar.markdown("---")
# --- END: "About Us" Section in Sidebar ---

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

embedding_model = load_embedding_model(model_name)

# --- File Processing & Dashboard Logic ---
if uploaded_files:
    new_data = []

    if not st.session_state.master_df.empty:
        existing_files = st.session_state.master_df['__file_name__'].tolist()
        files_to_process = [f for f in uploaded_files if f.name not in existing_files]
    else:
        files_to_process = uploaded_files

    if files_to_process:
        my_bar = st.progress(0, text="Initializing...")
        total_files = len(files_to_process)
        for i, file in enumerate(files_to_process):
            progress_stage1 = int(((i + 1) / total_files) * 70)
            my_bar.progress(progress_stage1, text=f"Step 1/3: Processing file {i+1}/{total_files} ({file.name})...")
            try:
                file_bytes = file.read()
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
                data = extract_metadata(text)
                data["__file_name__"] = file.name
                data["__file_bytes__"] = file_bytes
                new_data.append(data)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        if new_data:
            my_bar.progress(75, text="Step 2/3: Creating and enriching data...")
            new_df = pd.DataFrame(new_data)
            
            new_df['Consent Number'] = new_df['__file_name__'].str.replace('.pdf', '', case=False, regex=True).str.strip()

            new_df["GeoKey"] = new_df["Address"].str.lower().str.strip()
            new_df["Latitude"], new_df["Longitude"] = zip(*new_df["GeoKey"].apply(geocode_address))

            my_bar.progress(90, text="Step 3/3: Finalizing data...")
            auckland_tz = pytz.timezone("Pacific/Auckland")
            for col in ['Issue Date', 'Expiry Date']:
                # Ensure conversion to datetime objects for check_expiry function
                new_df[col] = pd.to_datetime(new_df[col], errors='coerce', dayfirst=True)
                # Localize only if timezone is not already present, otherwise astimezone
                new_df[col] = new_df[col].apply(localize_to_auckland)


            st.session_state.master_df = pd.concat([st.session_state.master_df, new_df], ignore_index=True)
            st.session_state.master_df.drop_duplicates(subset=['__file_name__'], keep='last', inplace=True)

            # Recalculate Consent Status Enhanced after new data is added and dates are processed
            df_to_update = st.session_state.master_df
            df_to_update["Consent Status Enhanced"] = df_to_update["Consent Status"] # Start with base status
            current_nz_aware_time = datetime.now(pytz.timezone("Pacific/Auckland"))
            
            # Ensure 'Expiry Date' is timezone-aware for comparison
            df_to_update['Expiry Date'] = df_to_update['Expiry Date'].apply(localize_to_auckland)

            expiring_mask = (
                (df_to_update["Consent Status"] == "Active") &
                (df_to_update["Expiry Date"].notna()) &
                (df_to_update["Expiry Date"] > current_nz_aware_time) &
                (df_to_update["Expiry Date"] <= current_nz_aware_time + timedelta(days=90))
            )
            df_to_update.loc[expiring_mask, "Consent Status Enhanced"] = "Expiring in 90 Days"


        corpus = st.session_state.master_df["Text Blob"].tolist()
        st.session_state.corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

        my_bar.progress(100, text="Processing Complete!")
        time.sleep(1)
        my_bar.empty()

# --- Main Dashboard Display ---
if not st.session_state.master_df.empty:
    df = st.session_state.master_df.copy()

    st.subheader("Consent Summary Metrics")
    col1, col2, col3, col4 = st.columns(4)
    color_map = {"Unknown": "gray", "Expired": "#8B0000", "Active": "green", "Expiring in 90 Days": "orange"}

    def colored_metric(column_obj, label, value, color):
        column_obj.markdown(f"""
            <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                <div style="font-size: 0.9em; color: #333;">{label}</div>
                <div style="font-size: 2.5em; font-weight: bold; color: {color};">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    colored_metric(col1, "Total Consents", len(df), "#4682B4")
    colored_metric(col2, "Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum(), color_map["Expiring in 90 Days"])
    colored_metric(col3, "Expired", df["Consent Status"].value_counts().get("Expired", 0), color_map["Expired"])
    colored_metric(col4, "Active", (df["Consent Status Enhanced"] == "Active").sum(), color_map["Active"])

    st.subheader("Consent Status Overview")
    status_counts = df["Consent Status Enhanced"].value_counts().reset_index(name="Count")
    status_counts.columns = ["Consent Status", "Count"]
    fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status", color_discrete_map=color_map, title="Consent Status Overview")
    fig_status.update_traces(textposition="outside")
    fig_status.update_layout(title_x=0.5)
    st.plotly_chart(fig_status, use_container_width=True)

    status_options = ["All"] + df["Consent Status Enhanced"].unique().tolist()
    status_filter = st.selectbox("Filter by Status (affects table and map below)", status_options)

    filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]

    with st.expander("Consent Table", expanded=True):
        columns_to_display = ["Consent Number", "Company Name", "Address", "Issue Date", "Expiry Date", "Consent Status Enhanced", "AUP(OP) Triggers"]
        
        display_df = filtered_df[columns_to_display].rename(columns={
            "Consent Status Enhanced": "Consent Status",
            "AUP(OP) Triggers": "AUP(OP) Triggers"
        })
        
        st.dataframe(display_df)
        csv_output = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Filtered CSV", csv_output, "filtered_consents.csv", "text/csv")

    with st.expander("Consent Map", expanded=True):
        map_df = filtered_df.dropna(subset=["Latitude", "Longitude"])
        if not map_df.empty:
            fig_map = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", hover_name="Company Name",
                                        hover_data={"Address": True, "Consent Status Enhanced": True, "Issue Date": True, "Expiry Date": True, "Consent Number": True},
                                        zoom=10, color="Consent Status Enhanced", color_discrete_map=color_map)
            fig_map.update_traces(marker=dict(size=12))
            fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No geocoded locations to display for the current filter.")

    st.markdown("#### **Historical View**")
    with st.expander("Expand to see consents issued per year", expanded=False):
        df_hist = df.copy()
        df_hist['Issue Year'] = df_hist['Issue Date'].dt.year
        issue_counts = df_hist['Issue Year'].value_counts().sort_index().reset_index(name='Number of Consents Issued')
        issue_counts.columns = ['Year', 'Number of Consents Issued']
        fig_hist = px.line(issue_counts, x='Year', y='Number of Consents Issued', markers=True, title='Consents Issued Per Year')
        fig_hist.update_traces(line=dict(color='#004489', width=3))
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### **Detailed Consent View**")
    with st.expander("Expand to select and view a specific consent", expanded=False):
        company_list = ["- Select a consent to view details -"] + sorted(df['Company Name'].unique().tolist())
        selected_company = st.selectbox("Select Consent by Company Name", options=company_list)
        if selected_company != "- Select a consent to view details -":
            detail_row = df[df['Company Name'] == selected_company].iloc[0]
            st.subheader(f"Details for: {detail_row['Company Name']}")
            st.markdown(f"**Resource Consent Number(s):** {detail_row['Resource Consent Numbers']}")
            st.markdown(f"**Address:** {detail_row['Address']}")
            st.markdown(f"**Issue Date:** {detail_row['Issue Date'].strftime('%d-%m-%Y') if pd.notna(detail_row['Issue Date']) else 'Unknown'}")
            st.markdown(f"**Expiry Date:** {detail_row['Expiry Date'].strftime('%d-%m-%Y') if pd.notna(detail_row['Expiry Date']) else 'Unknown'}")
            st.markdown(f"**Consent Status:** {detail_row['Consent Status Enhanced']}")
            st.markdown(f"**AUP(OP) Triggers:** {detail_row['AUP(OP) Triggers']}")
            st.markdown(f"**Reason for Consent:** {detail_row['Reason for Consent']}")
            st.markdown(f"**Consent Conditions:** {detail_row['Consent Conditions']}")

            # Provide PDF download link
            file_name_for_download = detail_row['__file_name__']
            file_bytes_for_download = detail_row['__file_bytes__']
            st.download_button(
                label=f"Download Original PDF: {file_name_for_download}",
                data=file_bytes_for_download,
                file_name=file_name_for_download,
                mime="application/pdf"
            )

    st.markdown("---")

    # --- LLM Chatbots ---
    st.subheader("AI Chatbots")
    tab_gemini, tab_groq, tab_semantic = st.tabs(["Gemini AI Chat", "Groq AI Chat (Langchain)", "LLM Semantic Query"])

    # --- Gemini AI Chat ---
with tab_gemini:
    if not google_api_key:
        st.warning("Google API Key not found. Please add it to your .env file or Streamlit secrets for Gemini AI functionality.")
    else:
        model = genai.GenerativeModel('gemini-pro')
        if "gemini_chat_history" not in st.session_state:
            st.session_state.gemini_chat_history = []

        for message in st.session_state.gemini_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["parts"])

        if prompt := st.chat_input("Ask Gemini about the consent data..."):
            st.session_state.gemini_chat_history.append({"role": "user", "parts": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- CORRECTED PREPARATION FOR MARKDOWN ---
            df_for_markdown_gemini = df.copy()
            # Iterate through columns and apply robust string conversion
            for col in df_for_markdown_gemini.columns:
                df_for_markdown_gemini[col] = df_for_markdown_gemini[col].apply(
                    lambda x: str(x).encode('utf-8', 'ignore').decode('utf-8') if pd.notna(x) else ''
                )
            # --- END CORRECTED PREPARATION ---

            context_for_gemini = "Here is the parsed consent data in a DataFrame format:\n"
            context_for_gemini += df_for_markdown_gemini.to_markdown(index=False) # Use the cleaned DataFrame
            context_for_gemini += "\n\nBased on this data, " + prompt

            with st.chat_message("assistant"):
                with st.spinner("Gemini is thinking..."):
                    try:
                        response = model.generate_content(context_for_gemini)
                        gemini_response = response.text
                        st.markdown(gemini_response)
                        st.session_state.gemini_chat_history.append({"role": "assistant", "parts": gemini_response})
                        log_ai_chat(prompt, gemini_response)
                    except Exception as e:
                        st.error(f"Gemini AI error: {e}")
                        st.session_state.gemini_chat_history.append({"role": "assistant", "parts": f"Sorry, I encountered an error: {e}"})

# --- Groq AI Chat ---
with tab_groq:
    if not groq_api_key:
        st.warning("Groq API Key not found. Please add it to your .env file or Streamlit secrets for Groq AI functionality.")
    else:
        if "groq_chat_history" not in st.session_state:
            st.session_state.groq_chat_history = []

        # Display chat messages from history on app rerun
        for message in st.session_state.groq_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask Groq about the consent data... ", key="groq_chat_input"):
            st.session_state.groq_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # --- CORRECTED PREPARATION FOR MARKDOWN ---
            df_for_markdown_groq = df.copy()
            # Iterate through columns and apply robust string conversion
            for col in df_for_markdown_groq.columns:
                df_for_markdown_groq[col] = df_for_markdown_groq[col].apply(
                    lambda x: str(x).encode('utf-8', 'ignore').decode('utf-8') if pd.notna(x) else ''
                )
            # --- END CORRECTED PREPARATION ---

            context_for_groq = "Here is the parsed consent data in a DataFrame format:\n"
            context_for_groq += df_for_markdown_groq.to_markdown(index=False)
            context_for_groq += "\n\nBased on this data, " + prompt

            # Initialize ChatGroq model
            chat = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="llama3-8b-8192")

            with st.chat_message("assistant"):
                with st.spinner("Groq is thinking..."):
                    try:
                        # Use Langchain's message format
                        messages = [
                            SystemMessage(content="You are a helpful AI assistant that answers questions based on provided consent data."),
                            HumanMessage(content=context_for_groq)
                        ]
                        response = chat.invoke(messages)
                        groq_response = response.content
                        st.markdown(groq_response)
                        st.session_state.groq_chat_history.append({"role": "assistant", "content": groq_response})
                        log_ai_chat(prompt, groq_response)
                    except Exception as e:
                        st.error(f"Groq AI error: {e}")
                        st.session_state.groq_chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {e}"})

    # --- LLM Semantic Query ---
    with tab_semantic:
        st.info("Use this to find relevant consent reports based on semantic similarity to your query.")

        if st.session_state.corpus_embeddings is not None:
            if st.session_state.semantic_search_query: 
                with st.spinner("Performing semantic search..."):
                    try:
                        query_embedding = embedding_model.encode(st.session_state.semantic_search_query, convert_to_tensor=True)
                        cosine_scores = util.cos_sim(query_embedding, st.session_state.corpus_embeddings)[0]
                        
                        # Get the top 5 most similar
                        top_results = st.session_state.master_df.copy()
                        top_results['score'] = cosine_scores.cpu().numpy()
                        top_results = top_results.sort_values(by='score', ascending=False).head(5)

                        st.subheader("Top 5 Semantically Similar Consent Reports:")
                        if not top_results.empty:
                            for index, row in top_results.iterrows():
                                with st.expander(f"**Consent:** {row['Company Name']} ({row['Consent Number']}) - Score: {row['score']:.4f}", expanded=False):
                                    st.markdown(f"**Address:** {row['Address']}")
                                    st.markdown(f"**Issue Date:** {row['Issue Date'].strftime('%d-%m-%Y') if pd.notna(row['Issue Date']) else 'Unknown'}")
                                    st.markdown(f"**Expiry Date:** {row['Expiry Date'].strftime('%d-%m-%Y') if pd.notna(row['Expiry Date']) else 'Unknown'}")
                                    st.markdown(f"**Consent Status:** {row['Consent Status Enhanced']}")
                                    st.markdown(f"**Reason for Consent (Snippet):** {row['Reason for Consent'][:500]}...") # Show a snippet
                                    st.markdown(f"**Consent Conditions (Snippet):** {row['Consent Conditions'][:500]}...") # Show a snippet

                                    file_name_sem = row['__file_name__']
                                    file_bytes_sem = row['__file_bytes__']
                                    st.download_button(
                                        label=f"Download Original PDF: {file_name_sem}",
                                        data=file_bytes_sem,
                                        file_name=file_name_sem,
                                        mime="application/pdf",
                                        key=f"download_sem_{index}"
                                    )
                        else:
                            st.info("No relevant results found.")
                    except Exception as e:
                        st.error(f"Semantic search error: {e}")
            else:
                st.info("Enter a query in the sidebar to perform a semantic search.")
        else:
            st.info("Please upload PDF files first to enable semantic search.")

    else:
        st.info("Upload PDF files using the sidebar to get started!")

# --- Chat Log Download ---
st.sidebar.markdown("---")
st.sidebar.markdown("#### Chat History")
chat_log_csv = get_chat_log_as_csv()
if chat_log_csv:
    st.sidebar.download_button(
        label="Download Chat Log (CSV)",
        data=chat_log_csv,
        file_name="ai_chat_log.csv",
        mime="text/csv"
    )
else:
    st.sidebar.info("No chat history to download yet.")

st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025")
