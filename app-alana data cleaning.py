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
# Streamlit Page Config & Style (MUST BE THE FIRST STREAMLIT COMMAND)
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿", initial_sidebar_state="expanded")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    # Display this warning once at startup if the key is missing
    st.error("Google API key not found. Gemini AI will be offline.")

# --- Weather Function ---
@st.cache_data(ttl=600)
def get_auckland_weather():
    if not openweathermap_api_key:
        return "Sunny, 18°C (offline mode)" # Default / fallback
    url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={openweathermap_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for HTTP errors
        data = response.json()
        if data.get("cod") != 200:
            return "Weather unavailable"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except requests.exceptions.RequestException:
        return "Weather unavailable (network error)"
    except Exception: # Catch other potential errors during JSON parsing, etc.
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
    <h1 style='color:#2c6e91; text-align:center; font-size:2.7em; font-family: Quicksand, sans-serif;'>
        Auckland Air Discharge Consent Dashboard
        
        Welcome to the Auckland Air Discharge Consent Dashboard. This dashboard allows you to upload the Air Discharge Resource Consent Decison Report to transforms your files into meaningful data.Add commentMore actions
        Explore the data by the csv file options or interact with the data using the Groq AI or LLM Semantic Query.
    </h1>
""", unsafe_allow_html=True)


# --- Utility Functions ---
def localize_to_auckland(dt):
    """
    Helper function to localize a datetime object to Pacific/Auckland timezone.
    Handles NaT and non-datetime types gracefully.
    """
    if pd.isna(dt) or not isinstance(dt, datetime):
        return pd.NaT # Return NaT if it's not a valid datetime

    auckland_tz = pytz.timezone("Pacific/Auckland")

    if dt.tzinfo is None:
        try:
            # Localize naive datetime. is_dst=None handles DST transitions by inferring or raising errors.
            return auckland_tz.localize(dt, is_dst=None)
        except pytz.AmbiguousTimeError:
            # For ambiguous times (e.g., during DST rollback), pick one (e.g., non-DST)
            # You might need a more specific business rule here.
            return auckland_tz.localize(dt, is_dst=False)
        except pytz.NonExistentTimeError:
            # For non-existent times (e.g., during DST spring forward), return NaT or adjust.
            return pd.NaT
    else:
        # If it's already timezone-aware, convert it to Auckland's timezone for consistency
        return dt.astimezone(auckland_tz)

def check_expiry(expiry_date):
    if pd.isna(expiry_date): # Handle missing data first
        return "Unknown"

    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland")) # Ensure the expiry date is timezone-aware for accurate comparison

    if expiry_date.tzinfo is None:
        try:
            localized_expiry_date = pytz.timezone("Pacific/Auckland").localize(expiry_date, is_dst=None)
        except pytz.AmbiguousTimeError:
            print(f"Warning: Ambiguous time for {expiry_date}. Defaulting to non-DST.")
            localized_expiry_date = pytz.timezone("Pacific/Auckland").localize(expiry_date, is_dst=False)
        except pytz.NonExistentTimeError:
            print(f"Warning: Non-existent time for {expiry_date}. Treating as Unknown.")
            return "Unknown" # handle specifically if you have a rule for non-existent times
        except Exception as e: # General fallback for other localization errors
            print(f"Warning: Could not localize expiry date {expiry_date}: {e}. Comparing as naive fallback (less robust).")
            return "Expired" if expiry_date < datetime.now(pytz.timezone("Pacific/Auckland")) else "Active"
    else:
        localized_expiry_date = expiry_date.astimezone(pytz.timezone("Pacific/Auckland"))

    return "Expired" if localized_expiry_date < current_nz_time else "Active"


@st.cache_data(show_spinner=False)
def geocode_address(address):
    # Normalize address: ensure it contains 'Auckland, New Zealand' for better geocoding accuracy
    standardized_address = address.strip()
    if not re.search(r'auckland', standardized_address, re.IGNORECASE):
        standardized_address += ", Auckland"
    if not re.search(r'new zealand|nz', standardized_address, re.IGNORECASE):
        standardized_address += ", New Zealand"

    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1) # Keep current rate limit

    try:
        location = geocode(standardized_address)
        if location:
            # print(f"DEBUG: Geocoded '{address}' -> ({location.latitude}, {location.longitude})") # Optional debug print
            return (location.latitude, location.longitude)
        else:
            print(f"DEBUG: Geocoding failed for '{standardized_address}' (no location found).") # Optional debug print
            return (None, None)
    except Exception as e:
        st.warning(f"Geocoding failed for '{standardized_address}': {e}")
        return (None, None)

def extract_metadata(text):
    # RC number patterns
    rc_patterns = [
        r"Application number:\s*(.+?)(?=\s*Applicant:)",
        r"Application numbers:\s*(.+?)(?=\s*Applicant:)",
        r"Application number(s):\s*(.+?)(?=\s*Applicant:)",
        r"Application number:\s*(.+?)(?=\s*Original consent)",
        r"Application numbers:\s*(.+?)(?=\s*Original consent)",
        r"Application number:\s*(.+?)(?=\s*Site address:)",
        r"Application numbers:\s*(.+?)(?=\s*Site address:)",
        r"Application number(s):\s*(.+?)(?=\s*Site address:)",
        r"RC[0-9]{5,}" 
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.DOTALL | re.MULTILINE | re.IGNORECASE))

    # Flatten list of lists/tuples that re.findall might return
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
        company_matches.extend(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
    company_str = ", ".join(list(dict.fromkeys(company_matches)))

    # Address patterns
    address_pattern = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_match = re.findall(address_pattern, text, re.MULTILINE | re.DOTALL)
    address_str = ", ".join(list(dict.fromkeys(address_match)))

    # Issue date patterns
    issue_date_patterns = [
        r"Commissioner\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"(\b\d{2}/\d{2}/\d{2}\b)"
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
                        if len(dt_str.split('/')[-1]) == 2:
                            issue_date = datetime.strptime(dt_str, "%d/%m/%y")
                        else:
                            issue_date = datetime.strptime(dt_str, "%d/%m/%Y")
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
        r"expire\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expires\s+on\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expires\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expire\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})",
        r"expires\s+([A-Za-z]+\s+years)",
        r"expire\s+([A-Za-z]+\s+years)",
        r"DIS\d{5,}(?:-w+)?\b\s+will\s+expire\s+(\d{1,}\s+years)",
        r"expires\s+(\d{1,}\s+months\s+[A-Za-z])+\s+[.?!]",
        r"expires\s+on\s+(\d{1,2}(?:st|nd|rd|th)\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expires\s+on\s+the\s+(\d{1,2}(?:st|nd|rd|th)\s+of\s+?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expire\s+on\s+(\d{1,2}/\d{1,2}/\d{4})",
        r"expire\s+on\s+(\d{1,2}-\d{1,2}-\d{4})",
        r"expire\s+([A-Za-z]+\s+(\d{1,})\s+years)",
        r"expire\s+(\d{1,2}\s+years)",
        r"expires\s+(\d{1,2}\s+years)",
        r"expire\s+([A-Za-z]+\s+(\d{1,2})\s+[A-Za-z]+)",
        r"earlier\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
        r"on\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"on\s+the\s+(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"(\d{1,}\s+years)",
    ]
    expiry_date = None
    expiry_date1 = []
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for dt_val_candidate in matches:
                dt_str = dt_val_candidate[0] if isinstance(dt_val_candidate, tuple) and dt_val_candidate else dt_val_candidate
                if not isinstance(dt_str, str) or not dt_str.strip():
                    continue

                try:
                    if '/' in dt_str:
                        if len(dt_str.split('/')[-1]) == 2:
                            expiry_date = datetime.strptime(dt_str, "%d/%m/%y")
                        else:
                            expiry_date = datetime.strptime(dt_str, "%d/%m/%Y")
                    else:

                        dt_str_cleaned = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?(?: of)?\b', r'\1', dt_str)
                        expiry_date = datetime.strptime(dt_str_cleaned, "%d %B %Y")

                    break

                except ValueError:
                    # If parsing fails for this match, just continue to the next one
                    continue
            if expiry_date:
                break
    
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                expiry_date1.append(" ".join([m for m in match if m]).strip())
            else:
                expiry_date1.append(match.strip())
    expiry_date1 = list(dict.fromkeys(expiry_date1))
    expiry_str= ", ".join(expiry_date1)
        
    # AUP triggers
    trigger_patterns = [
        r"(E14\.\d+\.\d+)",
        r"(E14\.\d+\.)",
        r"(NES:STO)",
        r"(NES:AQ)",
        r"(NES:IGHG)"
    ]
    triggers = []
    for pattern in trigger_patterns:
        triggers.extend(re.findall(pattern, text))
    triggers_str = " ".join(list(dict.fromkeys(triggers)))

    # Reason (Proposal)
    proposal_patterns= [
        r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\.)",
        r"Proposal\s*(.+?)(?=\n[A-Z]|\.)",
        r"Proposal\s*(.+?)(?=\n[A-Z]|\:)",
        r"Introduction and summary of proposal\s*(.+?)\s*Submissions",
        r"Proposal, site and locality description\s*(.+?)(?=\n[A-Z]|\.)",
        r"Summary of Decision\s*(.+?)(?=\n[A-Z]|\.)",
        r"Summary of proposal and activity status\s*(.+?)(?=\n[A-Z]|\.)"
    ]
    proposal =  []
    for pattern in proposal_patterns:
        proposal.extend(re.findall(pattern, text, re.MULTILINE | re.DOTALL))
    proposal_str= "".join(list(dict.fromkeys(proposal)))

    # Conditions (consolidated pattern for broader capture)
    conditions_patterns = [
        r"(?:Specific conditions - Air Discharge DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions -)",
        r"(?:Air Quality conditions).*?(?=Wastewater Discharge conditions)",
        r"(?:Air Discharge Permit Conditions).*?(?=E\. Definitions)",
        r"(?:Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=DIS\d{5,}(?:-\w+)?\b)",
        r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?:Conditions Specific to air quality).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:regional discharge DIS\d{5,}(?:-w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?:Specific conditions - Air Discharge DIS\d{5,}\b).*?(?=Advice notes)",
        r"(?:Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=Advice notes)",
        r"(?:DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=DIS\d{5,}(?:-\w+)?\b - Specific conditions)",
        r"(?:Specific Conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?:Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b Only).*?(?=Advice notes)",
        r"(?:Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Specific Conditions -)",
        r"(?:SPECIFIC CONDITIONS - DISCHARGE TO AIR DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Conditions relevant to Discharge Permit DIS\d{5,}(?:-\w+)?\b only).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit (DIS\d{5,}(?:-\w+)?\b)).*?(?=Advice notes)",
        r"(?:Specific conditions - DIS\d{5,}(?:-\w+)?\b (air)).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specifc conditions)",
        r"(?:Attachment 1: Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?:Specific Air Discharge Conditions).*?(?=Advice notes)",
        r"(?:Specific conditions - Discharge to Air: DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - discharge permit (air discharge) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Air Discharge Limits).*?(?= Acoustic Conditions)",
        r"(?:Specific conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit (s15) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Secific conditions)",
        r"(?:Specific conditions relating to Air discharge permit - DIS\d{5,}(?:-\w+)?\b).*?(?=General Advice notes)",
        r"(?:Specific conditions - Discharge permit (s15) - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:Specific Conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?:Specific conditions - Discharge to air: DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?:Attachement 1: Consolidated conditions of consent as amended).*?(?=Resource Consent Notice of Works Starting)",
        r"(?:Specific conditions - Air Discharge consent - DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?:Specific conditions - Discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?:DIS\d{5,}(?:-\w+)?\b - Air Discharge).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?:DIS\d{5,}(?:-\w+)?\b & DIS\d{5,}(?:-\w+)?\b).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?:Specific conditions - Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice Notes - General)",
        r"(?:AIR QUALITY - ROCK CRUSHER).*?(?=GROUNDWATER)",
        # Fallback broad pattern if specific ones fail
        r"(?<=Conditions).*?(?=Advice notes)"
    ]

    conditions_str = []
    for pattern in conditions_patterns:
        conditions_match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        if conditions_match:
            conditions_str = conditions_match.group(0).strip()
            break

    # Extracting numbered conditions (if conditions_str is found)
    conditions_numbers = []
    if conditions_str:
        temp_conditions_matches = re.findall(r"^\s*(\d+\.?\d*)\s*[A-Z].*?(?=\n\s*\d+\.?\d*\s*[A-Z]|\Z)", conditions_str, re.MULTILINE | re.DOTALL)
        flattened_temp_conditions = []
        for item in temp_conditions_matches:
            if isinstance(item, tuple):
                flattened_temp_conditions.append(item[0])
            else:
                flattened_temp_conditions.append(item)

        conditions_numbers = [re.match(r'^(\d+\.?\d*)', cn.strip()).group(1) for cn in flattened_temp_conditions if isinstance(cn, str) and re.match(r'^(\d+\.?\d*)', cn.strip())]
        conditions_numbers = list(dict.fromkeys(conditions_numbers))        

    return {
        "Resource Consent Numbers": rc_str if rc_str else "Unknown Resource Consent Numbers",
        "Company Name": company_str if company_str else "Unknown Company Name",
        "Address": address_str if address_str else "Unknown Address",
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown Issue Date",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else expiry_str,
        "AUP(OP) Triggers": triggers_str if triggers_str else "Unknown AUP Triggers",
        "Reason for Consent": proposal_str if proposal_str else "Unknown Reason for Consent",
        "Consent Condition Numbers": ", ".join(conditions_numbers) if conditions_numbers else "Unknown Condition Numbers",
        "Consent Status": check_expiry(expiry_date), # This will now use the localized date
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
            if df_log.empty:
                return None
            output = io.StringIO()
            df_log.to_csv(output, index=False)
            return output.getvalue().encode("utf-8")
        except pd.errors.EmptyDataError:
            return None
        except Exception as e:
            st.error(f"Error reading chat log: {e}")
            return None
    return None


# --- Sidebar & Model Loader ---
st.sidebar.markdown("""
    <h2 style='color:#2c6e91; font-family:Segoe UI, Roboto, sans-serif;'>
        Control Panel
    </h2>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("Choose Embedding Model:", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("LLM Semantic Search Query")

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

embedding_model = load_embedding_model(model_name)

# --- CACHING EMBEDDINGS FOR PERFORMANCE ---
@st.cache_data(show_spinner="Generating document embeddings...")
def get_corpus_embeddings(text_blobs_tuple, model_name_str):
    """Generates and caches embeddings for all text blobs."""
    model_obj=load_embedding_model(model_name_str)
    return model_obj.encode(list(text_blobs_tuple), convert_to_tensor=True)

df = pd.DataFrame()

# --- File Processing & Dashboard ---
if uploaded_files:
    # --- START: MULTI-STAGE PROGRESS BAR ---
    my_bar = st.progress(0, text="Initializing...")
    all_data = []
    total_files = len(uploaded_files)

    # Stage 1: PDF Processing (0% -> 70% of total progress)
    for i, file in enumerate(uploaded_files):
        # Calculate progress within the 0-70 range
        progress_stage1 = int(((i + 1) / total_files) * 70)
        my_bar.progress(progress_stage1, text=f"Step 1/3: Processing file {i+1}/{total_files} ({file.name})...")
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
    # --- END Stage 1 ---

    if all_data:
        # --- Stage 2: Geocoding (70% -> 90% of total progress) ---
        my_bar.progress(75, text="Step 2/3: Geocoding addresses. This may take a moment...")
        df = pd.DataFrame(all_data)
        df["GeoKey"] = df["Address"].str.lower().str.strip()
        # Geocoding is the slow part of this stage
        df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))

        # --- Stage 3: Finalizing and Rendering (90% -> 100%) ---
        my_bar.progress(90, text="Step 3/3: Finalizing data and rendering dashboard...")

        # --- DATETIME LOCALIZATION ---
        auckland_tz = pytz.timezone("Pacific/Auckland")
        df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce', dayfirst=True)
        df['Expiry Date'] = df['Expiry Date'].apply(localize_to_auckland)
        
        # --- ENHANCED STATUS CALCULATION ---
        df["Consent Status Enhanced"] = df["Consent Status"]
        current_nz_aware_time = datetime.now(pytz.timezone("Pacific/Auckland"))
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"] > current_nz_aware_time) &
            (df["Expiry Date"] <= current_nz_aware_time + timedelta(days=90)),
            "Consent Status Enhanced"
        ] = "Expiring in 90 Days"

        # --- START RENDERING DASHBOARD ---
        
        # Metrics
        st.subheader("Consent Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)

        def colored_metric(column_obj, label, value, color):
            """Displays a metric value with a custom color using markdown."""
            column_obj.markdown(f"""
                <div style="text-align: center; padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin-bottom: 10px;">
                    <div style="font-size: 0.9em; color: #333;">{label}</div>
                    <div style="font-size: 2.5em; font-weight: bold; color: {color};">{value}</div>
                </div>
            """, unsafe_allow_html=True)

        color_map = {"Unknown": "gray", "Expired": "#8B0000", "Active": "green", "Expiring in 90 Days": "orange"}
        total_consents = len(df)
        expiring_90_days = (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum()
        expired_count = df["Consent Status"].value_counts().get("Expired", 0)
        truly_active_count = (df["Consent Status Enhanced"] == "Active").sum()

        colored_metric(col1, "Total Consents", total_consents, "#4682B4")
        colored_metric(col2, "Expiring in 90 Days", expiring_90_days, color_map["Expiring in 90 Days"])
        colored_metric(col3, "Expired", expired_count, color_map["Expired"])
        colored_metric(col4, "Active", truly_active_count, color_map["Active"])

        # Status Chart
        status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts.columns = ["Consent Status", "Count"]
        fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status", color_discrete_map=color_map)
        fig_status.update_traces(textposition="outside")
        fig_status.update_layout(title="Consent Status Overview", title_x=0.5)
        st.plotly_chart(fig_status, use_container_width=True)

        # Consent Table
        with st.expander("Consent Table", expanded=True):
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist())
            filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
            display_df = filtered_df[[
                "__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date",
                "Consent Status Enhanced", "AUP(OP) Triggers", "Reason for Consent", "Consent Condition Numbers"
            ]].rename(columns={"__file_name__": "File Name", "Consent Status Enhanced": "Consent Status"})
            st.dataframe(display_df)
            csv_output = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_output, "filtered_consents.csv", "text/csv")

        # Consent Map
        with st.expander("Consent Map", expanded=True):
            map_df = df.dropna(subset=["Latitude", "Longitude"])
            if not map_df.empty:
                fig = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", hover_name="Company Name",
                                        hover_data={"Address": True, "Consent Status Enhanced": True, "Issue Date": True, "Expiry Date": True},
                                        zoom=10, color="Consent Status Enhanced", color_discrete_map=color_map)
                fig.update_traces(marker=dict(size=12))
                fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

        # Semantic Search
        with st.expander("LLM Semantic Search Results", expanded=True):
            if query_input:
                corpus = df["Text Blob"].tolist()
                corpus_embeddings = get_corpus_embeddings(tuple(corpus), model_name)
                query_embedding = embedding_model.encode(query_input, convert_to_tensor=True)
                scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_k_indices = scores.argsort(descending=True)

                displayed_results = 0
                similarity_threshold = st.slider("LLM Semantic Search Relevance Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

                for idx in top_k_indices:
                    score = scores[idx.item()]
                    if score < similarity_threshold and displayed_results >= 1:
                        break
                    row = df.iloc[idx.item()]
                    st.markdown(f"**{displayed_results + 1}. {row['Company Name']} - {row['Address']}** (Similarity: {score:.2f})")
                    st.markdown(f"- **Triggers**: {row['AUP(OP) Triggers']}")
                    expiry_display = row['Expiry Date'].strftime('%Y-%m-%d') if pd.notna(row['Expiry Date']) else 'N/A'
                    st.markdown(f"- **Expires**: {expiry_display}")
                    safe_filename = clean_surrogates(row['__file_name__'])
                    st.download_button(label=f"Download PDF ({safe_filename})", data=row['__file_bytes__'], file_name=safe_filename, mime="application/pdf", key=f"download_{idx.item()}")
                    st.markdown("---")
                    displayed_results += 1
                    if displayed_results >= 3:
                        break
                if displayed_results == 0:
                    st.info(f"No highly relevant documents found for your query with a similarity score above {similarity_threshold:.2f}.")
        
        # --- END RENDERING DASHBOARD ---

        # Finalize and remove the progress bar
        my_bar.progress(100, text="Dashboard Ready!")
        time.sleep(1)
        my_bar.empty()

    else: # Handle case where no data could be extracted from any files
        my_bar.empty()
        st.warning("Could not extract any data from the uploaded files. Please check the file contents.")


# ----------------------------
# Ask AI About Consents Chatbot
# ----------------------------

st.markdown("---") # Horizontal line for separation
st.subheader("Ask AI About Consents")

with st.expander("AI Chatbot", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, consent conditions, or general trends)", unsafe_allow_html=True)

    llm_provider = st.radio("Choose LLM Provider", ["Groq AI", "Gemini AI"], horizontal=True, key="llm_provider_radio")
    chat_input = st.text_area("Search any query:", key="chat_input_text_area")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Ask AI", key="ask_ai_button"):
        if not chat_input.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    # ==================================================================
                    # MODIFICATION START: Implement RAG (Option 3)
                    # ==================================================================
                    context_sample_list = []

                    if not df.empty:
                        # 1. Perform Semantic Search to find relevant documents for the chat query
                        corpus = df["Text Blob"].tolist()
                        corpus_embeddings = get_corpus_embeddings(tuple(corpus), model_name)
                        query_embedding = embedding_model.encode(chat_input, convert_to_tensor=True)
                        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

                        # 2. Get top 5 most relevant document indices
                        top_k = 5
                        top_k_indices = scores.argsort(descending=True)[:top_k]
                        
                        # 3. Build a small, highly-relevant context from the search results
                        relevant_docs_data = []
                        for idx in top_k_indices:
                            # Only include if the relevance score is decent (e.g., > 0.3)
                            if scores[idx.item()] > 0.3:
                                relevant_docs_data.append(df.iloc[idx.item()])

                        if not relevant_docs_data:
                            st.warning("Could not find any documents relevant to your query. The AI will answer without specific context.")
                            context_sample_list = []
                        else:
                            st.info(f"Found {len(relevant_docs_data)} relevant documents. Providing them to the AI as context.")
                            context_df = pd.DataFrame(relevant_docs_data)
                            context_sample_df = context_df[[
                                "Company Name", "Resource Consent Numbers","Address", "Consent Status", "AUP(OP) Triggers",
                                "Consent Conditions", "Issue Date", "Expiry Date", "Reason for Consent"
                            ]].dropna().copy()

                            # Format datetime columns for JSON serialization
                            for col in ['Expiry Date', 'Issue Date']:
                                if col in context_sample_df.columns and pd.api.types.is_datetime64_any_dtype(context_sample_df[col]):
                                    context_sample_df[col] = context_sample_df[col].dt.strftime('%Y-%m-%d %H:%M:%S %Z%z')
                            
                            context_sample_list = context_sample_df.to_dict(orient="records")
                    
                    else:
                        st.info("No documents uploaded. AI is answering with general knowledge or default sample data.")
                        # Use a default sample if no files are uploaded
                        context_sample_list = [{"Company Name": "Default Sample Ltd", "Resource Consent Numbers": "DIS60327400", "Address": "123 Default St, Auckland", "Consent Status": "Active", "AUP(OP) Triggers": "E14.1.1 (default)", "Consent Conditions": "Consent Conditions", "Issue Date": "2024-01-01", "Expiry Date": "2025-12-31", "Reason for Consent": "General default operations"}]
                    
                    context_sample_json = json.dumps(context_sample_list, indent=2)
                    
                    # ==================================================================
                    # MODIFICATION END
                    # ==================================================================

                    current_auckland_time_str = datetime.now(pytz.timezone("Pacific/Auckland")).strftime("%Y-%m-%d")

                    system_message_content = f"""
                    You are an intelligent assistant specializing in Auckland Air Discharge Consents. Your core task is to answer user questions exclusively and precisely using the "Provided Data" below.

                    Crucial Directives:
                    1.  **Strict Data Adherence:** Base your entire response solely on the information contained within the 'Provided Data'. Do not introduce any external knowledge, assumptions, or speculative content.
                    2.  **Direct Retrieval:** Prioritize direct retrieval of facts from the 'Provided Data'. When answering about locations, refer to the 'Address' field.
                    3.  **Handling Missing Information/Complex Analysis:** If the answer to any part of the user's query cannot be directly found or calculated from the 'Provided Data' *as presented*, or if it requires complex analysis/aggregation of data not explicitly shown (e.g., counting items not in the top 5, or performing complex filtering across a large dataset), you *must* explicitly state: "I cannot find that information within the currently uploaded documents, or it requires more complex analysis than I can perform with the provided data. Please refer to the dashboard's tables and filters for detailed insights."
                    4.  **Current Date Context:** The current date in Auckland for reference is {current_auckland_time_str}. Use this if the query relates to the current status or remaining time for consents.
                    5.  **Concise Format:** Present your answer in clear, concise bullet points.
                    6.  **Tone:** Maintain a helpful, professional, and purely data-driven tone.

                    ---
                    Provided Data (JSON format):
                    """

                    user_query = f"""
{system_message_content}
{context_sample_json}

---
User Query: {chat_input}

Answer:
"""

                    answer_raw = ""
                    if llm_provider == "Gemini":
                        if google_api_key:
                            GEMINI_MODEL_TO_USE = "models/gemini-1.0-pro"
                            gemini_model = genai.GenerativeModel(GEMINI_MODEL_TO_USE)
                            try:
                                response = gemini_model.generate_content(user_query)
                                if response and hasattr(response, 'text'):
                                    answer_raw = response.text
                                else:
                                    answer_raw = "Gemini generated an empty or invalid response. It might have been filtered for safety reasons or encountered an internal error. Check your console for details."
                            except Exception as e:
                                answer_raw = f"Gemini API error: {e}. This could be due to the chosen Gemini model ('{GEMINI_MODEL_TO_USE}') not being available or an API issue."
                        else:
                            answer_raw = "Gemini AI is offline (Google API key not found)."
                    
                    elif llm_provider == "Groq AI": # FIX: Changed from "Groq" to "Groq AI"
                        if groq_api_key:
                            chat_groq = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
                            try:
                                groq_response = chat_groq.invoke([
                                    SystemMessage(content=system_message_content + "\n" + context_sample_json),
                                    HumanMessage(content=f"User Query: {chat_input}")
                                ])
                                answer_raw = groq_response.content if hasattr(groq_response, 'content') else str(groq_response)
                            except Exception as e:
                                answer_raw = f"Groq API error: {e}"
                        else:
                            answer_raw = "Groq AI is offline (Groq API key not found)."
                    else: 
                        st.warning("Selected LLM provider is not available or supported.")
                        answer_raw = "AI provider not available."


                    st.markdown(f"### 🖥️  Answer from {llm_provider}\n\n{answer_raw}")

                    if answer_raw and "offline" not in answer_raw and "unavailable" not in answer_raw and "API error" not in answer_raw and "Gemini API error" not in answer_raw:
                        log_ai_chat(chat_input, answer_raw)

                except Exception as e:
                    st.error(f"AI interaction error: {e}")

    chat_log_csv = get_chat_log_as_csv()
    if chat_log_csv:
        st.download_button(
            label="Download Chat History (CSV)",
            data=chat_log_csv,
            file_name="ai_chat_history.csv",
            mime="text/csv",
            help="Download a CSV file containing all past AI chat interactions."
        )
    else:
        st.info("No chat history available yet.")

st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025")
