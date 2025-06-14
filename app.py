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

# --- LLM Specific Imports ---
import google.generativeai as genai # For Gemini
from openai import OpenAI # For OpenAI (new client)
from langchain_groq import ChatGroq # For Groq (Langchain integration)
from langchain_core.messages import SystemMessage, HumanMessage # Needed for Langchain messages
# --- End LLM Specific Imports ---

# --- API Key Setup ---
load_dotenv()
# Prioritize st.secrets for deployment, fall back to .env for local development
openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
# OpenWeatherMap API key
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

# --- DEBUGGING API KEY LOADING (prints to console, not Streamlit UI) ---
print(f"DEBUG: OpenAI API Key Loaded: {bool(openai_api_key)}")
print(f"DEBUG: Groq API Key Loaded: {bool(groq_api_key)}")
print(f"DEBUG: Google API Key Loaded: {bool(google_api_key)}")
print(f"DEBUG: OpenWeatherMap API Key Loaded: {bool(openweathermap_api_key)}")
# --- END DEBUGGING API KEY LOADING ---


# ------------------------
# Streamlit Page Config & Style (MUST BE THE FIRST STREAMLIT COMMAND)
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

# Initialize clients and models (after set_page_config)
client = OpenAI(api_key=openai_api_key) if openai_api_key else None

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
    </h1>
""", unsafe_allow_html=True)


# --- Utility Functions ---
def check_expiry(expiry_date):
    """
    Checks the expiry status of a consent, handling timezone-aware comparisons.
    Assumes expiry_date from PDFs are naive and represent local Auckland time.
    """
    if expiry_date is None:
        return "Unknown"
    
    # Get current Auckland time, which is timezone-aware
    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
    
    # Localize the naive expiry_date to Auckland timezone for a valid comparison.
    # If the expiry_date somehow already has timezone info, convert it to Auckland's timezone.
    if expiry_date.tzinfo is None:
        try:
            localized_expiry_date = pytz.timezone("Pacific/Auckland").localize(expiry_date)
        except Exception as e:
            # Fallback if localization fails (e.g., ambiguous time during DST changes)
            # In production, you might want more robust handling or logging here.
            print(f"Warning: Could not localize expiry date {expiry_date}: {e}. Comparing as naive.")
            return "Expired" if expiry_date < datetime.now() else "Active" # Fallback to naive local comparison
    else:
        # If it's already timezone-aware, convert it to Auckland's timezone for consistency
        localized_expiry_date = expiry_date.astimezone(pytz.timezone("Pacific/Auckland"))

    return "Expired" if localized_expiry_date < current_nz_time else "Active"


@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    try:
        location = geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except Exception as e:
        st.warning(f"Geocoding failed for '{address}': {e}")
        return (None, None)

def extract_metadata(text):
    # RC number patterns
    rc_patterns = [
        r"Application number:\s*(.+?)(?=\s*Applicant)",
        r"Application numbers:\s*(.+)(?=\s*Applicant)",
        r"Application number(?:s)?:\s*(.+)(?=\s*Applicant)", # MODIFIED: non-capturing group for 's'
        r"RC[0-9]{5,}" # Added fallback for RC numbers
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Flatten list of lists/tuples that re.findall might return
    flattened_rc_matches = []
    for item in rc_matches:
        if isinstance(item, tuple):
            flattened_rc_matches.append(item[-1]) # Take the last element of the tuple, usually the actual RC number
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
        company_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    company_str = ", ".join(list(dict.fromkeys(company_matches)))

    # Address patterns
    address_pattern = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_match = re.findall(address_pattern, text, re.IGNORECASE)
    address_str = ", ".join(list(dict.fromkeys(address_match)))

    # Issue date patterns
    issue_date_patterns = [
        r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"(\b\d{2}/\d{2}/\d{2}\b)" # Specific pattern for dd/mm/yy
    ]
    issue_date = None
    for pattern in issue_date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for dt_str in matches:
                if isinstance(dt_str, tuple):
                    dt_str = dt_str[0] if dt_str else ""
                if not isinstance(dt_str, str) or not dt_str:
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
        r"expire on (\d{1,2} [A-Za-z]+ \d{4})",
        r"expires on (\d{1,2} [A-Za-z]+ \d{4})",
        r"expires (\d{1,2} [A-Za-z]+ \d{4})",
        r"expire (\d{1,2} [A-Za-z]+ \d{4})",
        r"(\d{1,} years) from the date of commencement",
        r"DIS\d{5,}(?:-\w+)?\b will expire (\d{1,} years [A-Za-z]+[.?!])",
        r"expires (\d{1,} months [A-Za-z]+)[.?!]",
        r"expires (\d{1,} years [A-Za-z]+)[.?!]",
        r"expire on (\d{1,2}/\d{1,2}/\d{4})",
        r"expire ([A-Za-z]\d{1,} years)",
    ]
    expiry_date = None
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for dt_val in matches:
                dt_str = dt_val[0] if isinstance(dt_val, tuple) and dt_val else dt_val
                if not isinstance(dt_str, str) or not dt_str:
                    continue

                try:
                    if '/' in dt_str:
                        expiry_date = datetime.strptime(dt_str, "%d/%m/%Y")
                    elif re.match(r'^\d{1,2} [A-Za-z]+ \d{4}$', dt_str):
                        dt_str = re.sub(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', r'\1', dt_str)
                        expiry_date = datetime.strptime(dt_str, "%d %B %Y")
                    else:
                        years_match = re.search(r'(\d+)\s*years', dt_str, re.IGNORECASE)
                        months_match = re.search(r'(\d+)\s*months', dt_str, re.IGNORECASE)

                        if years_match and issue_date:
                            num_years = int(years_match.group(1))
                            expiry_date = issue_date + timedelta(days=num_years * 365)
                        elif months_match and issue_date:
                            num_months = int(months_match.group(1))
                            expiry_date = issue_date + timedelta(days=num_months * 30)
                        else:
                            continue
                    break
                except ValueError:
                    continue
            if expiry_date:
                break

    # AUP triggers
    trigger_patterns = [
        r"(E14\.\d+\.\d+)",
        r"(E14\.\d+\.)",
        r"(NES:STO)",
        r"(NES:AQ)"
    ]
    triggers = []
    for pattern in trigger_patterns:
        triggers.extend(re.findall(pattern, text))
    triggers_str = " ".join(list(dict.fromkeys(triggers)))

    # Reason (Proposal)
    proposal_pattern = r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\.)"
    proposal_matches = re.findall(proposal_pattern, text, re.DOTALL)
    proposal_str = " ".join(list(dict.fromkeys(proposal_matches)))

    # Conditions (consolidated pattern for broader capture)
    conditions_patterns = [
        r"(?:Conditions).*?(?=Advice notes)",
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
        r"(?:Conditions\n).*?(?=(?:Advice notes|Schedule \d+|APPENDIX \w+|E\. Definitions|\Z))",
    ]

    conditions_str = ""
    for pattern in conditions_patterns:
        conditions_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
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

    # Management Plans from conditions
    managementplan_raw = r"(?i)\b(\w+)\sManagement Plan"
    management_plan = re.findall(managementplan_raw, conditions_str, re.DOTALL)
    managementplan_final = list(dict.fromkeys([f"{word} Management Plan" for word in management_plan]))

    return {
        "Resource Consent Numbers": rc_str if rc_str else "Unknown",
        "Company Name": company_str if company_str else "Unknown",
        "Address": address_str if address_str else "Unknown",
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown",
        "AUP(OP) Triggers": triggers_str if triggers_str else "None",
        "Reason for Consent": proposal_str if proposal_str else "Unknown",
        "Consent Conditions": ", ".join(conditions_numbers) if conditions_numbers else "None",
        "Mitigation (Consent Conditions)": ", ".join(managementplan_final) if managementplan_final else "None",
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
query_input = st.sidebar.text_input("Semantic Search Query")

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

embedding_model = load_embedding_model(model_name)

# Initialize df outside the if block to ensure it always exists
df = pd.DataFrame()

# --- File Processing & Dashboard ---
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
        auckland_tz = pytz.timezone("Pacific/Auckland")
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True)
        def localize_to_auckland(dt):
            if pd.isna(dt): # Handle NaT values
                return pd.NaT
            if dt.tzinfo is None:
                try:
                    # Using 'infer_dst=True' helps with Daylight Saving Time transitions
                    return auckland_tz.localize(dt, is_dst=None) # is_dst=None lets pytz infer or raise Ambiguous/NonExistentTimeError
                except pytz.AmbiguousTimeError:
                    # Fallback for ambiguous times (e.g., during DST rollback)
                    # You might need a more specific strategy here, like choosing 'infer_dst=True' or 'fold=True' in localize
                    # For simplicity, returning a naive datetime in such rare cases might be acceptable or logging it.
                    return auckland_tz.localize(dt, is_dst=False) # Or True, depending on desired resolution
                except pytz.NonExistentTimeError:
                    # Fallback for non-existent times (e.g., during DST spring forward)
                    return pd.NaT # Treat as invalid or make a sensible adjustment
            return dt.astimezone(auckland_tz) # Convert if already timezone-aware but different TZ
        df['Expiry Date'] = df['Expiry Date'].apply(localize_to_auckland)
        

        df["Consent Status Enhanced"] = df["Consent Status"]
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"] > datetime.now(pytz.timezone("Pacific/Auckland"))) & # Use timezone-aware current time
            (df["Expiry Date"] <= datetime.now(pytz.timezone("Pacific/Auckland")) + timedelta(days=90)), # Use timezone-aware current time
            "Consent Status Enhanced"
        ] = "Expiring in 90 Days"

        # Metrics
        st.subheader("Consent Summary Metrics")
        col1, col2, col3, col4 = st.columns(4) # Added a 4th column for "Truly Active"
        col1.metric("Total Consents", len(df))
        col2.metric("Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum())
        col3.metric("Expired", df["Consent Status"].value_counts().get("Expired", 0))
        # Calculate "Truly Active" (Active, not expiring soon, and not Unknown)
        truly_active_count = (df["Consent Status Enhanced"] == "Active").sum()
        col4.metric("Truly Active", truly_active_count)


        # Status Chart
        status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts.columns = ["Consent Status", "Count"]
        color_map = {"Unknown": "gray", "Expired": "red", "Active": "green", "Expiring in 90 Days": "orange"}
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
                "Consent Status Enhanced", "AUP(OP) Triggers", "Reason for Consent", "Mitigation (Consent Conditions)"
            ]].rename(columns={
                "__file_name__": "File Name",
                "Consent Status Enhanced": "Consent Status"
            })
            st.dataframe(display_df)
            csv_output = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_output, "filtered_consents.csv", "text/csv")
            
        # Consent Map
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
                        "Consent Status Enhanced": True,
                        "Issue Date": True,
                        "Expiry Date": True
                    },
                    zoom=10,
                    color="Consent Status Enhanced",
                    color_discrete_map=color_map
                )
                fig.update_traces(marker=dict(size=12))
                fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            
        # Semantic Search
        with st.expander("Semantic Search Results", expanded=True):
            if query_input:
                corpus = df["Text Blob"].tolist()
                corpus_embeddings = embedding_model.encode(corpus, convert_to_tensor=True) # Use embedding_model
                query_embedding = embedding_model.encode(query_input, convert_to_tensor=True) # Use embedding_model
                scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_k = scores.argsort(descending=True)[:3]
                for i, idx in enumerate(top_k):
                    row = df.iloc[idx.item()]
                    st.markdown(f"**{i+1}. {row['Company Name']} - {row['Address']}**")
                    st.markdown(f"- **Triggers**: {row['AUP(OP) Triggers']}")
                    st.markdown(f"- **Expires**: {row['Expiry Date']}")
                    safe_filename = clean_surrogates(row['__file_name__'])
                    st.download_button(label=f"Download PDF ({safe_filename})", data=row['__file_bytes__'], file_name=safe_filename, mime="application/pdf", key=f"download_{i}")
                    st.markdown("---")

# ----------------------------
# Ask AI About Consents Chatbot
# ----------------------------

st.markdown("---") # Horizontal line for separation
st.subheader("Ask AI About Consents")

with st.expander("AI Chatbot", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)

    llm_provider = st.radio("Choose LLM Provider", ["Gemini", "OpenAI", "Groq"], horizontal=True, key="llm_provider_radio")
    chat_input = st.text_area("Search any query:", key="chat_input_text_area")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Ask AI", key="ask_ai_button"):
        if not chat_input.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    context_sample_list = []
                    
                    if not df.empty:
                        context_sample_df = df[[
                            "Company Name", "Resource Consent Numbers","Address", "Consent Status", "AUP(OP) Triggers",
                            "Mitigation (Consent Conditions)", "Issue Date", "Expiry Date", "Reason for Consent"
                        ]].dropna().copy()
                        
                        for col in ['Expiry Date', 'Issue Date']:
                            if col in context_sample_df.columns and pd.api.types.is_datetime64_any_dtype(context_sample_df[col]):
                                context_sample_df[col] = context_sample_df[col].dt.strftime('%Y-%m-%d')
                                
                        context_sample_list = context_sample_df.to_dict(orient="records")
                    else:
                        st.info("No documents uploaded. AI is answering with general knowledge or default sample data.")
                        context_sample_list = [{"Company Name": "Default Sample Ltd", "Resource Consent Numbers": "DIS60327400", "Address": "123 Default St, Auckland", "Consent Status": "Active", "AUP(OP) Triggers": "E14.1.1 (default)", "Mitigation (Consent Conditions)": "General Management Plan", "Issue Date": "2024-01-01", "Expiry Date": "2025-12-31", "Reason for Consent": "General default operations"}]

                    context_sample_raw_json = json.dumps(context_sample_list, indent=2)
                    context_sample_json = "" # Will store the potentially truncated JSON

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

                    full_query_for_token_check = system_message_content + context_sample_raw_json + f"\n--- \nUser Query: {chat_input}\n\nAnswer:"
                    
                    MAX_TOKENS_FOR_PROMPT = 30000 
                    
                    if llm_provider == "Gemini" and google_api_key:
                        try:
                            # ### IMPORTANT: REPLACE WITH THE EXACT MODEL NAME YOU FOUND IN YOUR CONSOLE OUTPUT! ###
                            # Common options based on your list: "models/gemini-1.0-pro", "models/gemini-1.5-pro-latest", "models/gemini-flash-latest"
                            GEMINI_MODEL_TO_USE = "models/gemini-1.0-pro" # <-- CHANGE THIS LINE BASED ON YOUR CONSOLE OUTPUT
                            
                            temp_model_for_token_count = genai.GenerativeModel(GEMINI_MODEL_TO_USE) 
                            token_count_response = temp_model_for_token_count.count_tokens(full_query_for_token_check)
                            total_tokens = token_count_response.total_tokens

                            if total_tokens > MAX_TOKENS_FOR_PROMPT:
                                st.warning("The uploaded data is very large. Attempting to reduce context for AI.")
                                num_entries_to_send = min(len(context_sample_list), 400) 

                                context_sample_json = json.dumps(context_sample_list[:num_entries_to_send], indent=2)
                                st.info(f"Reduced context to approximately {num_entries_to_send} entries due to potential token limits.")
                            else:
                                context_sample_json = context_sample_raw_json
                        except Exception as e:
                            st.warning(f"Could not count tokens for Gemini: {e}. Sending full data (may exceed limits). This could be due to the chosen Gemini model not being available or an API issue. **Verify the model name ('{GEMINI_MODEL_TO_USE}') in your code matches an available model from your console output.**")
                            context_sample_json = context_sample_raw_json 
                    else:
                        if len(context_sample_raw_json) > 80000: 
                            st.warning("The uploaded data is very large. Only a portion will be sent to the AI to prevent exceeding token limits.")
                            num_entries_to_send = min(len(context_sample_list), 100)
                            context_sample_json = json.dumps(context_sample_list[:num_entries_to_send], indent=2)
                        else:
                            context_sample_json = context_sample_raw_json

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
                            # ### IMPORTANT: REPLACE WITH THE EXACT MODEL NAME YOU FOUND IN YOUR CONSOLE OUTPUT! ###
                            # Common options based on your list: "models/gemini-1.0-pro", "models/gemini-1.5-pro-latest", "models/gemini-flash-latest"
                            GEMINI_MODEL_TO_USE = "models/gemini-1.0-pro" # <-- CHANGE THIS LINE BASED ON YOUR CONSOLE OUTPUT
                            
                            gemini_model = genai.GenerativeModel(GEMINI_MODEL_TO_USE) 
                            try:
                                response = gemini_model.generate_content(user_query)
                                if response and hasattr(response, 'text'):
                                    answer_raw = response.text
                                else:
                                    answer_raw = "Gemini generated an empty or invalid response. It might have been filtered for safety reasons or encountered an internal error. Check your console for details."
                            except Exception as e:
                                answer_raw = f"Gemini API error: {e}. This could be due to the chosen Gemini model ('{GEMINI_MODEL_TO_USE}') not being available or an API issue. **Verify the model name in your code matches an available model from your console output.**"
                        else:
                            answer_raw = "Gemini AI is offline (Google API key not found)."
                    elif llm_provider == "OpenAI":
                        if client:
                            messages = [
                                {"role": "system", "content": system_message_content + "\n" + context_sample_json},
                                {"role": "user", "content": f"User Query: {chat_input}"}
                            ]
                            try:
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=messages,
                                    max_tokens=500,
                                    temperature=0.7
                                )
                                answer_raw = response.choices[0].message.content
                            except Exception as e:
                                answer_raw = f"OpenAI API error: {e}"
                        else:
                            answer_raw = "OpenAI AI is offline (OpenAI API key not found)."
                    elif llm_provider == "Groq":
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

                    st.markdown(f"### 🖥️  Answer from {llm_provider} AI\n\n{answer_raw}")
                    
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
