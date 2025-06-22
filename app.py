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
from langchain_groq import ChatGroq # For Groq (Langchain integration)
from langchain_core.messages import SystemMessage, HumanMessage # Needed for Langchain messages
# --- End LLM Specific Imports ---

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

# ------------------------
# Streamlit Page Config & Style (MUST BE THE FIRST STREAMLIT COMMAND)
# ------------------------
# UPDATED: Added initial_sidebar_state="expanded"
st.set_page_config(
    page_title="Auckland Air Discharge Consent Dashboard",
    layout="wide",
    page_icon="🇳🇿",
    initial_sidebar_state="expanded" 
)

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
            return auckland_tz.localize(dt, is_dst=False) 
        except pytz.NonExistentTimeError:
            # For non-existent times (e.g., during DST spring forward), return NaT or adjust.
            return pd.NaT 
    else:
        # If it's already timezone-aware, convert it to Auckland's timezone for consistency
        return dt.astimezone(auckland_tz)

def check_expiry(expiry_date):
    if pd.isna(expiry_date): # Ensure this check remains first for NaT from pd.to_datetime
        return "Unknown"
    
    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland")) # <-- Define this consistently

    # Ensure expiry_date is timezone-aware before comparison
    if expiry_date.tzinfo is None:
        localized_expiry_date = localize_to_auckland(expiry_date)
        if pd.isna(localized_expiry_date):
             # Fallback for unlocalizable dates
            return "Expired" if expiry_date < datetime.now() else "Active"
    else:
        localized_expiry_date = expiry_date.astimezone(pytz.timezone("Pacific/Auckland"))

    return "Expired" if localized_expiry_date < current_nz_time else "Active"


@st.cache_data(show_spinner=False)
def geocode_address(address):
    # Normalize address
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
        else:
            return (None, None)
    except Exception as e:
        st.warning(f"Geocoding failed for '{standardized_address}': {e}")
        return (None, None)

def extract_metadata(text):
    # RC number patterns
    rc_patterns = [
        r"Application number:\s*(.+?)\s*Applicant",
        r"Application numbers:\s*(.+)\s*Applicant",
        r"Application number\(s\):\s*(.+)\s*Applicant",
        r"Application number:\s*(.+)\s*Original consent",
        r"Application numbers:\s*(.+)\s*Original consent",
        r"RC[0-9]{5,}" 
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.MULTILINE |re.IGNORECASE | re.DOTALL))
    
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
        company_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    company_str = ", ".join(list(dict.fromkeys(company_matches)))

    # Address patterns
    address_pattern = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_match = re.findall(address_pattern, text, re.MULTILINE | re.IGNORECASE)
    address_str = ", ".join(list(dict.fromkeys(address_match)))

    # Issue date patterns
    issue_date_patterns = [
        r"Commissioner\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)"
    ]
    
    # --- BUG FIX ---
    # The empty, non-functional if/else block that printed to the screen has been removed.
    
    issue_date = None   
    for pattern in issue_date_patterns:
        matches = re.findall(pattern, text)
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
                    continue
            if expiry_date:
                break

    # ... (Rest of the function is the same)
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
        proposal.extend(re.findall(pattern, text))
    proposal_str= "".join(list(dict.fromkeys(proposal)))

    # Conditions
    conditions_patterns = [r"(?<=Conditions).*?(?=Advice notes)"] # Simplified for brevity, original list was extensive
    conditions_str = ""
    for pattern in conditions_patterns:
        conditions_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if conditions_match:
            conditions_str = conditions_match.group(0).strip()
            break
    
    return {
        "Resource Consent Numbers": rc_str if rc_str else "Unknown",
        "Company Name": company_str if company_str else "Unknown",
        "Address": address_str if address_str else "Unknown",
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown",
        "AUP(OP) Triggers": triggers_str if triggers_str else "Unknown",
        "Reason for Consent": proposal_str if proposal_str else "Unknown",
        "Consent Conditions": conditions_str if conditions_str else "Unknown",
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

# --- CACHING EMBEDDINGS FOR PERFORMANCE ---
@st.cache_data(show_spinner="Generating document embeddings...")
def get_corpus_embeddings(text_blobs_tuple, model_name_str):
    """Generates and caches embeddings for all text blobs."""
    model_obj=load_embedding_model(model_name_str)
    return model_obj.encode(list(text_blobs_tuple), convert_to_tensor=True)

df = pd.DataFrame()

# --- File Processing & Dashboard ---
if uploaded_files:
    all_data = []
    with st.spinner("Processing PDF files..."):
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
        
        # --- Data Cleaning and Transformation ---
        with st.spinner("Analyzing and geocoding data..."):
            df["GeoKey"] = df["Address"].str.lower().str.strip()
            df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))
            
            df['Issue Date'] = pd.to_datetime(df['Issue Date'], errors='coerce', dayfirst=True)
            df['Expiry Date'] = pd.to_datetime(df['Expiry Date'], errors='coerce', dayfirst=True)
            
            df['Issue Date'] = df['Issue Date'].apply(localize_to_auckland)
            df['Expiry Date'] = df['Expiry Date'].apply(localize_to_auckland)

            df["Consent Status Enhanced"] = df['Expiry Date'].apply(check_expiry)
            
            current_nz_aware_time = datetime.now(pytz.timezone("Pacific/Auckland"))
            ninety_days_from_now = current_nz_aware_time + timedelta(days=90)

            expiring_mask = (df["Consent Status Enhanced"] == "Active") & \
                            (df["Expiry Date"].notna()) & \
                            (df["Expiry Date"] <= ninety_days_from_now)
            df.loc[expiring_mask, "Consent Status Enhanced"] = "Expiring in 90 Days"

        # --- Dashboard Display ---
        st.subheader("Consent Summary Metrics")
        col1, col2, col3, col4 = st.columns(4) 
        
        status_counts = df["Consent Status Enhanced"].value_counts()
        total_consents = len(df)
        expiring_90_days = status_counts.get("Expiring in 90 Days", 0)
        expired_count = status_counts.get("Expired", 0)
        active_count = status_counts.get("Active", 0)

        col1.metric("Total Consents", total_consents)
        col2.metric("Active", active_count)
        col3.metric("Expiring in 90 Days", expiring_90_days)
        col4.metric("Expired", expired_count)

        # Status Chart
        color_map = {"Unknown": "gray", "Expired": "#d9534f", "Active": "#5cb85c", "Expiring in 90 Days": "orange"}
        status_counts_df = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts_df.columns = ["Status", "Count"]
        fig_status = px.bar(status_counts_df, x="Status", y="Count", text="Count", color="Status", color_discrete_map=color_map)
        fig_status.update_layout(title="Consent Status Overview", title_x=0.5, xaxis_title=None, yaxis_title="Number of Consents")
        st.plotly_chart(fig_status, use_container_width=True)

        # Consent Table
        with st.expander("Consent Table", expanded=True):
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist())
            filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
            
            # Create a display version with formatted dates
            display_df = filtered_df.copy()
            for col in ['Issue Date', 'Expiry Date']:
                 if col in display_df.columns and pd.api.types.is_datetime64_any_dtype(display_df[col]):
                    display_df[col] = display_df[col].dt.strftime('%d-%b-%Y')

            st.dataframe(display_df[[
                "__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date",
                "Consent Status Enhanced"
            ]].rename(columns={
                "__file_name__": "File Name",
                "Consent Status Enhanced": "Consent Status"
            }))
            csv_output = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download as CSV", csv_output, "filtered_consents.csv", "text/csv")
            
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
                fig.update_layout(mapbox_style="open-street-map", margin={"r":
