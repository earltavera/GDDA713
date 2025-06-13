# Auckland Air Discharge Consent Dashboard - Complete with Gemini, OpenAI, and Groq Chatbot + Data Cleaning

import streamlit as st
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

import pandas as pd
import pymupdf as fitz # Alias pymupdf to fitz for consistency with original code
import re
from datetime import datetime, timedelta
import plotly.express as px
from sentence_transformers import SentenceTransformer, util
# Removed: from geopy.geolocators import Nominatim
# Removed: from geopy.extra.rate_limiter import RateLimiter
import os
from dotenv import load_dotenv
import csv
import io
import requests
import pytz
from openai import OpenAI
import google.generativeai as genai
from langchain_groq import ChatGroq
from pdf2image import convert_from_bytes
import pytesseract
from dateutil.relativedelta import relativedelta # Added for precise date calculations
import time # Added for manual rate limiting

# --------------------
# Load Environment Variables
# --------------------

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY") # Ensure OpenAI API key is loaded
google_api_key = os.getenv("GOOGLE_API_KEY") # Ensure Google API key is loaded

client = OpenAI(api_key=openai_api_key) # Initialize OpenAI client with key
genai.configure(api_key=google_api_key) # Configure Gemini with key

# --------------------
# Weather Function
# --------------------
@st.cache_data(ttl=600)
def get_auckland_weather():
    """Fetches current weather data for Auckland using OpenWeatherMap API."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather unavailable (API key missing)"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={api_key}"
        response = requests.get(url, timeout=5) # Added a timeout for robustness
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except requests.exceptions.Timeout:
        return "Weather unavailable (request timed out)"
    except requests.exceptions.RequestException as e:
        # Catch specific request exceptions (e.g., network issues)
        st.error(f"Weather API request error: {e}")
        return "Weather unavailable (network error)"
    except (KeyError, IndexError) as e:
        # Catch errors if expected keys are missing from the JSON response
        st.error(f"Weather data parsing error: {e}")
        return "Weather data unavailable (parsing error)"
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"An unexpected error occurred while fetching weather: {e}")
        return "Weather unavailable"

# --------------------
# Utility Functions
# --------------------

def check_expiry(expiry_date):
    """Checks the status of a consent based on its expiry date."""
    if expiry_date is None or pd.isna(expiry_date): # Also check for NaT from pandas
        return "Unknown"
    # Ensure comparison is timezone-aware if expiry_date has tz info
    return "Expired" if expiry_date < datetime.now().date() else "Active" # Compare dates only

@st.cache_data(show_spinner=False)
def geocode_address_direct_nominatim(address):
    """
    Geocodes a given address to latitude and longitude using Nominatim (OpenStreetMap) direct API.
    Includes basic error handling for network issues and missing data.
    """
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": address,
        "format": "json",
        "limit": 1,
        "user-agent": "air_discharge_dashboard_direct_api" # Important for Nominatim's usage policy
    }
    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        if data and len(data) > 0:
            latitude = data[0].get("lat")
            longitude = data[0].get("lon")
            if latitude is not None and longitude is not None:
                return (float(latitude), float(longitude))
        return (None, None) # Return None, None if no data or lat/lon missing
    except requests.exceptions.Timeout:
        st.warning(f"Geocoding request timed out for '{address}'.")
        return (None, None)
    except requests.exceptions.RequestException as e:
        st.warning(f"Geocoding failed for '{address}' (Network/HTTP error): {e}")
        return (None, None)
    except Exception as e:
        st.warning(f"An unexpected error occurred during geocoding for '{address}': {e}")
        return (None, None)
    
def parse_mixed_date(date_str):
    """
    Parses a date string from various common formats.
    Returns a datetime.date object or None if parsing fails.
    """
    if not isinstance(date_str, str): # Handle non-string inputs
        return None
    
    formats = [
        "%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y", # Common formats
        "%d %b %Y", "%d %B, %Y", # e.g., "1 Jan, 2023"
        "%Y-%m-%d" # ISO format
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date() # Return date only
        except (ValueError, TypeError):
            continue
    return None
    
# --------------------
# Data Cleaning & Extraction
# --------------------

def extract_metadata(text):
    """
    Extracts various metadata fields from the raw text of a PDF consent document.
    Includes logic for relative expiry dates.
    """
    data = {
        "Resource Consent Numbers": "Unknown",
        "Company Name": "Unknown",
        "Address": "Unknown",
        "Issue Date": "Unknown",
        "Expiry Date": "Unknown",
        "AUP(OP) Triggers": "Unknown",
        "Reason for Consent": "Unknown",
        "Consent Conditions": "Unknown",
        "Mitigation (Consent Conditions)": "Unknown",
        "Text Blob": text # Keep text blob for debugging
    }

    # RC number patterns
    rc_raw_patterns = [
        r"Application number(?:s)?:\s*(.+?)(?=\s*Applicant|\n)",
        r"Application number\(s\):\s*(.+?)(?=\s*Applicant|\n)",
        r"RC[0-9]{5,}(?:-[A-Z]{1,4})?" # e.g., RC12345-AIR
    ]
    rc_matches = []
    for pattern in rc_raw_patterns:
        rc_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    if rc_matches:
        # Filter for actual RC numbers if broad patterns capture too much
        rc_numbers_found = [re.search(r"RC[0-9]{5,}(?:-[A-Z]{1,4})?", m).group(0) for m in rc_matches if re.search(r"RC[0-9]{5,}(?:-[A-Z]{1,4})?", m)]
        data["Resource Consent Numbers"] = ", ".join(list(dict.fromkeys(rc_numbers_found))) if rc_numbers_found else "Unknown"

    # Company name patterns
    company_raw_patterns = [
        r"Applicant:\s*(.+?)(?=\s*Site address|\n)",
        r"Applicant's name:\s*(.+?)(?=\s*Site address|\n)"
    ]
    company_matches = []
    for pattern in company_raw_patterns:
        company_matches.extend(re.findall(pattern, text, re.DOTALL | re.IGNORECASE))
    if company_matches:
        data["Company Name"] = " ".join(list(dict.fromkeys([m.strip() for m in company_matches])))

    # Address pattern
    address_raw_pattern = r"Site address:\s*(.+?)(?=\s*Legal description|\n)"
    address_matches = re.findall(address_raw_pattern, text, re.DOTALL | re.IGNORECASE)
    if address_matches:
        data["Address"] = " ".join(list(dict.fromkeys([m.strip() for m in address_matches])))

    # Issue date patterns
    issue_date_raw_patterns = [
        r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date of decision:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date of issue:\s*(\d{1,2} [A-Za-z]+ \d{4})"
    ]
    issue_date = None
    for pattern in issue_date_raw_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            issue_date = parse_mixed_date(match.group(1))
            if issue_date:
                data["Issue Date"] = issue_date.strftime("%d-%m-%Y")
                break

    # Expiry date patterns and calculation (Enhanced)
    expiry_date = None
    expiry_raw_patterns = [
        r"expire(?:s)? on (\d{1,2} [A-Za-z]+ \d{4})",
        r"expire(?:s)? on (\d{1,2}/\d{1,2}/\d{2,4})",
        r"expire(?:s)? (\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)", # Matches "expires 1st January 2025"
        r"(\d+)\s*(year|month)s? from the date of commencement", # Catches "5 years from the date of commencement"
        r"DIS\d{5,}(?:-\w+)?\b will expire (\d+)\s*(year|month)s?", # Catches "DIS12345 will expire 10 years"
        r"expires (\d+)\s*(year|month)s?", # Catches "expires 5 years"
        r"expire (\d+)\s*(year|month)s?" # Catches "expire 3 months"
    ]

    for pattern in expiry_raw_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # Check if it's a direct date first
            is_direct_date = re.match(r"\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}", match.group(1))
            if is_direct_date:
                parsed_date = parse_mixed_date(match.group(1))
                if parsed_date:
                    expiry_date = parsed_date
                    break
            else: # Must be a relative duration (e.g., "5 years")
                if len(match.groups()) >= 2: # Check if unit is captured (e.g., 'year', 'month')
                    try:
                        value = int(match.group(1))
                        unit = match.group(2).lower()
                        if issue_date: # Need an issue date to calculate relative expiry
                            if "year" in unit:
                                expiry_date = issue_date + relativedelta(years=value)
                            elif "month" in unit:
                                expiry_date = issue_date + relativedelta(months=value)
                            if expiry_date: # If calculated, break
                                break
                    except ValueError:
                        pass # Not a number, continue to next pattern

    if expiry_date:
        data["Expiry Date"] = expiry_date.strftime("%d-%m-%Y")
    else:
        # Fallback to general capture if no specific pattern matched a date/duration
        general_expiry_patterns = [
            r"will expire(?:s)? on?\s*(.+?)(?=\s*\(|;|\n)", # Catches "will expire on 1 Jan 2025 (" or "will expire 5 years;"
            r"expiry date:\s*(.+?)(?=\s*\n)",
            r"term of consent is\s*(.+?)(?=\s*\n)",
            r"duration of consent is\s*(.+?)(?=\s*\n)"
        ]
        for pattern in general_expiry_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                potential_date_str = match.group(1).strip()
                # Try parsing as a direct date
                potential_expiry_date = parse_mixed_date(potential_date_str)
                if potential_expiry_date:
                    expiry_date = potential_expiry_date
                    data["Expiry Date"] = expiry_date.strftime("%d-%m-%Y")
                    break
                else:
                    # Try to find relative durations in the general text
                    duration_match = re.search(r"(\d+)\s*(year|month)s?", potential_date_str, re.IGNORECASE)
                    if duration_match and issue_date:
                        try:
                            value = int(duration_match.group(1))
                            unit = duration_match.group(2).lower()
                            if "year" in unit:
                                expiry_date = issue_date + relativedelta(years=value)
                            elif "month" in unit:
                                expiry_date = issue_date + relativedelta(months=value)
                            if expiry_date:
                                data["Expiry Date"] = expiry_date.strftime("%d-%m-%Y")
                                break
                        except ValueError:
                            pass

    # AUP triggers
    trigger_raw_patterns = [
        r"(E14\.\d+\.\d+)", r"(E14\.\d+\.)", r"(NES:STO)", r"(NES:AQ)",
        r"(Chapter E14: Air Quality)", r"(National Environmental Standards for Air Quality)"
    ]
    trigger_matches = []
    for pattern in trigger_raw_patterns:
        trigger_matches.extend(re.findall(pattern, text, re.IGNORECASE))
    if trigger_matches:
        data["AUP(OP) Triggers"] = ", ".join(list(dict.fromkeys([m.strip() for m in trigger_matches])))

    # Proposal / Reason for Consent (more flexible regex)
    proposal_raw_patterns = [
        r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\n\d+\.|\Z)", # Matches until a new section header or end of document
        r"Description of Activity:\s*(.+?)(?=\n[A-Z]|\n\d+\.|\Z)",
        r"Purpose of consent:\s*(.+?)(?=\n[A-Z]|\n\d+\.|\Z)"
    ]
    proposal_matches = []
    for pattern in proposal_raw_patterns:
        proposal_matches.extend(re.findall(pattern, text, re.DOTALL | re.IGNORECASE))
    if proposal_matches:
        data["Reason for Consent"] = " ".join(list(dict.fromkeys([m.strip() for m in proposal_matches])))

    # Consent Conditions (extracting full text of conditions block)
    # Using more general patterns for conditions blocks
    conditions_raw_patterns = [
        r"Conditions\s*(.*?)(?=Advice notes|General Advice notes|E\. Definitions|Resource Consent Notice of Works Starting|\Z)",
        r"Specific conditions - Air Discharge(?:.*?)?\b(.*?)(?=Specific conditions -|\Z)",
        r"Air Quality conditions\s*(.*?)(?=Wastewater Discharge conditions|Advice notes|\Z)",
        r"Air Discharge Permit Conditions\s*(.*?)(?=E\. Definitions|Advice notes|\Z)",
        r"SPECIFIC CONDITIONS - DISCHARGE TO AIR\s*(.*?)(?=Advice notes|\Z)",
        r"Conditions relevant to Air Discharge Permit(?:.*?)\b(?:Only)?\s*(.*?)(?=Advice notes|Specific Conditions -|\Z)",
        r"Specific Air Discharge Conditions\s*(.*?)(?=Advice notes|\Z)",
        r"Air Discharge Limits\s*(.*?)(?=Acoustic Conditions|\Z)",
        r"Consolidated conditions of consent as amended\s*(.*?)(?=Advice notes|\Z)"
    ]
    conditions_full_text = []
    for pattern in conditions_raw_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            conditions_full_text.append(match.group(1).strip())
    
    if conditions_full_text:
        # Clean up bullet points/numbering and join
        cleaned_conditions = re.sub(r"^\d+\.\s*", "", "\n".join(conditions_full_text), flags=re.MULTILINE)
        cleaned_conditions = re.sub(r"^\s*-\s*", "", cleaned_conditions, flags=re.MULTILINE)
        data["Consent Conditions"] = cleaned_conditions
        
        # Management Plans within the extracted conditions
        management_plan = re.findall(r"(?i)\b(\w+)\sManagement Plan", data["Consent Conditions"])
        if management_plan:
            data["Mitigation (Consent Conditions)"] = ", ".join(list(dict.fromkeys([f"{word} Management Plan" for word in management_plan])))

    # Calculate Consent Status
    if data["Expiry Date"] != "Unknown":
        # Parse the extracted expiry date string back to datetime.date for check_expiry
        parsed_expiry_date_for_status = parse_mixed_date(data["Expiry Date"])
        data["Consent Status"] = check_expiry(parsed_expiry_date_for_status)
    else:
        data["Consent Status"] = "Unknown"


    return data


def clean_surrogates(text):
    """Encodes and decodes text to remove surrogate characters that can cause issues."""
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer_raw):
    """Logs AI chat interactions to a CSV file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer_raw}
    file_path = "ai_chat_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Timestamp", "Question", "Answer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def get_chat_log_as_csv():
    """Retrieves the AI chat log as a CSV string."""
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

# --------------------
# PDF Processing Function (Cached for Performance)
# --------------------

@st.cache_data(show_spinner="Processing PDF(s)... This might take a moment.", persist="disk") # Persist to disk for larger files
def process_uploaded_pdfs(uploaded_files_list):
    """
    Processes a list of uploaded PDF files, extracts metadata, and returns a list of dictionaries.
    This function is cached to prevent re-processing the same files on every Streamlit rerun.
    """
    all_data = []
    for file in uploaded_files_list:
        try:
            file_bytes = file.getvalue()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)

            # OCR Fallback if text extraction yields little to no content
            if not text.strip() or len(text.strip()) < 100: # Added a length check for robustness
                # IMPORTANT: pytesseract requires the Tesseract OCR engine to be installed
                # on your system and its path configured (e.g., via pytesseract.pytesseract.tesseract_cmd)
                # This part will fail if Tesseract is not installed and configured.
                if 'convert_from_bytes' in globals() and 'pytesseract' in globals():
                    st.warning(f"{file.name} appears to be image-based or has minimal text. Attempting OCR...")
                    try:
                        # Convert all pages to images (can be memory intensive for large PDFs)
                        images = convert_from_bytes(file_bytes)
                        ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
                        if ocr_text.strip():
                            text = ocr_text # Use OCR text if successful
                            st.success(f"OCR successfully extracted text from {file.name}.")
                        else:
                            st.error(f"OCR failed to extract readable text from {file.name}. Raw text might be empty.")
                    except Exception as ocr_e:
                        st.error(f"OCR processing failed for {file.name}: {ocr_e}")
                        # Keep original (potentially empty) text
                else:
                    st.warning(f"{file.name} appears to be image-based, but OCR libraries (pdf2image, pytesseract) might not be fully configured or Tesseract OCR engine is missing. Text extraction might be incomplete.")

            data = extract_metadata(text)
            data["__file_name__"] = file.name
            data["__file_bytes__"] = file_bytes # Store bytes for later download
            all_data.append(data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}. Please ensure it's a valid PDF.")
    return all_data

# --------------------
# Banner
# --------------------

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

# --------------------
# Sidebar & Model Loader
# --------------------

st.sidebar.markdown("""
    <h2 style='color:#2c6e91; font-family:Segoe UI, Roboto, sans-serif;'>
        Control Panel
    </h2>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("Choose Semantic Search Model:", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query (e.g., 'dust management plan for construction')")

@st.cache_resource(show_spinner="Loading semantic search model...")
def load_model(name):
    """Loads a SentenceTransformer model for semantic search."""
    return SentenceTransformer(name)

model = load_model(model_name)

# ------------------------
# File Processing & Dashboard Display
# ------------------------
if uploaded_files:
    # Call the cached function to process PDFs
    all_data = process_uploaded_pdfs(uploaded_files)

    if all_data:
        # Create DataFrame from processed data
        df = pd.DataFrame(all_data)

        # Normalize dates after loading into DataFrame
        df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors='coerce', dayfirst=True).dt.date
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True).dt.date

        # Apply geocoding (this will be done once per unique address due to caching)
        df["GeoKey"] = df["Address"].astype(str).str.lower().str.strip()
        unique_addresses = df["GeoKey"].unique()
        
        geocoded_results = {}
        for i, addr in enumerate(unique_addresses):
            # Implement manual rate limiting for Nominatim (1 request per second)
            time.sleep(1) 
            geocoded_results[addr] = geocode_address_direct_nominatim(addr)
            # Add a progress indicator for long geocoding tasks
            if i % 10 == 0: # Update every 10 addresses
                st.sidebar.progress((i + 1) / len(unique_addresses), text=f"Geocoding addresses: {i+1}/{len(unique_addresses)}")

        df["Latitude"] = df["GeoKey"].apply(lambda x: geocoded_results.get(x, (None, None))[0])
        df["Longitude"] = df["GeoKey"].apply(lambda x: geocoded_results.get(x, (None, None))[1])


        df["Consent Status Enhanced"] = df["Consent Status"]
        
        # Identify consents expiring in the next 90 days
        current_date = datetime.now().date()
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"].notna()) & # Ensure Expiry Date is not NaT
            (df["Expiry Date"] >= current_date) & # Must be active or expiring
            (df["Expiry Date"] <= current_date + timedelta(days=90)),
            "Consent Status Enhanced"
        ] = "Expiring Soon (90 Days)"
        
        # Re-evaluate "Expired" status using the date objects for precision
        df.loc[
            (df["Expiry Date"].notna()) & # Ensure Expiry Date is not NaT
            (df["Expiry Date"] < current_date),
            "Consent Status Enhanced"
        ] = "Expired"

        # Metrics
        st.subheader("Consent Summary Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Consents", len(df))
        col2.metric("Expired", (df["Consent Status Enhanced"] == "Expired").sum())
        col3.metric("Expiring Soon", (df["Consent Status Enhanced"] == "Expiring Soon (90 Days)").sum())

        # Status Chart
        status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts.columns = ["Consent Status", "Count"]
        color_map = {"Unknown": "gray", "Expired": "red", "Active": "green", "Expiring Soon (90 Days)": "orange"}
        fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status", color_discrete_map=color_map)
        fig_status.update_traces(textposition="outside")
        fig_status.update_layout(title="Consent Status Overview", title_x=0.5)
        st.plotly_chart(fig_status, use_container_width=True)

        # Consent Table
        with st.expander("Consent Table", expanded=True):
            st.info("💡 **Debugging Tip:** Click 'View Raw Text & Data' below each row to inspect extracted text and data for missing fields.")
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist(), key="table_status_filter")
            filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
            
            # Prepare DataFrame for display and CSV export
            display_df = filtered_df[[
                "__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date",
                "Consent Status Enhanced", "AUP(OP) Triggers", "Reason for Consent", "Mitigation (Consent Conditions)"
            ]].rename(columns={
                "__file_name__": "File Name",
                "Consent Status Enhanced": "Consent Status"
            })
            
            # Display the DataFrame row by row with expanders
            for index, row in display_df.iterrows():
                # Display core information for the row
                st.markdown(f"**{row['File Name']}** - {row['Company Name']}")
                st.markdown(f"**Consent No:** {row['Resource Consent Numbers']} | **Address:** {row['Address']}")
                st.markdown(f"**Issue Date:** {row['Issue Date']} | **Expiry Date:** {row['Expiry Date']} | **Status:** {row['Consent Status']}")
                st.markdown(f"**Triggers:** {row['AUP(OP) Triggers']}")
                st.markdown(f"**Reason:** {row['Reason for Consent']}")
                st.markdown(f"**Mitigation:** {row['Mitigation (Consent Conditions)']}")

                # Debug expander
                with st.expander(f"View Raw Text & Data for {row['File Name']}", expanded=False):
                    original_entry = next((item for item in all_data if item['__file_name__'] == row['File Name']), None)
                    if original_entry:
                        st.markdown("**Extracted Text Blob:**")
                        st.text_area(f"Full Text for {row['File Name']}", original_entry.get('Text Blob', 'No text extracted'), height=300)
                        
                        st.markdown("**Raw Extracted Data:**")
                        debug_data = {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v for k, v in original_entry.items() if k not in ['Text Blob', '__file_bytes__']}
                        st.json(debug_data)
                    else:
                        st.info("Original data entry not found for debugging.")
                
                # Download button for the current row's PDF
                safe_filename = clean_surrogates(row['File Name'])
                original_bytes = next((item['__file_bytes__'] for item in all_data if item['__file_name__'] == row['File Name']), None)
                if original_bytes:
                    st.download_button(
                        label=f"📄 Download Original PDF: {safe_filename}",
                        data=original_bytes,
                        file_name=safe_filename,
                        mime="application/pdf",
                        key=f"download_pdf_{index}"
                    )
                st.markdown("---") # Separator between entries

            # CSV Download Button (for the full filtered display_df)
            if not display_df.empty:
                csv_data = display_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Filtered Consents CSV",
                    csv_data,
                    "filtered_consents.csv",
                    "text/csv",
                    key="download_filtered_csv"
                )


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
                        "Issue Date": "|%d-%m-%Y", # Format date for hover
                        "Expiry Date": "|%d-%m-%Y" # Format date for hover
                    },
                    zoom=10,
                    height=500,
                    color="Consent Status Enhanced",
                    color_discrete_map=color_map
                )
                fig.update_traces(marker=dict(size=12))
                fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No geocodable addresses found to display on the map.")

        # Enhanced Semantic Search Section
        with st.expander("Semantic Search Results", expanded=True):
            if query_input:
                st.info("Running enhanced semantic + structured search...")

                def normalize_text_for_search(text):
                    """Normalizes text for consistent embedding and search."""
                    return re.sub(r"\s+", " ", str(text).lower())

                # Create a comprehensive corpus for semantic search
                corpus = (
                    df["Company Name"].fillna("") + " | " +
                    df["Address"].fillna("") + " | " +
                    df["AUP(OP) Triggers"].fillna("") + " | " +
                    df["Mitigation (Consent Conditions)"].fillna("") + " | " +
                    df["Reason for Consent"].fillna("") + " | " +
                    df["Consent Conditions"].fillna("") + " | " + # Include full conditions for search
                    df["Resource Consent Numbers"].fillna("") + " | " +
                    df["Text Blob"].fillna("") # Full text blob provides richest context
                ).apply(normalize_text_for_search).tolist()

                query_input_norm = normalize_text_for_search(query_input)
                corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
                query_embedding = model.encode(query_input_norm, convert_to_tensor=True)

                if "e5" in model_name or "bge" in model_name:
                    scores = query_embedding @ corpus_embeddings.T
                else:
                    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

                # Get top K results based on semantic similarity
                top_k = scores.argsort(descending=True)[:5] # Get indices of top 5 results

                # Also perform a simple keyword match fallback
                keyword_matches_df = df[
                    df["Address"].astype(str).str.contains(query_input, case=False, na=False) |
                    df["Resource Consent Numbers"].astype(str).str.contains(query_input, case=False, na=False) |
                    df["Reason for Consent"].astype(str).str.contains(query_input, case=False, na=False) |
                    df["Company Name"].astype(str).str.contains(query_input, case=False, na=False) |
                    df["Consent Conditions"].astype(str).str.contains(query_input, case=False, na=False)
                ]

                # Display results
                # Use a threshold for semantic similarity to only show meaningful results
                if top_k is not None and any(scores[top_k] > 0.4): # Increased threshold slightly for quality
                    st.success("Top semantic results:")
                    for i, idx in enumerate(top_k):
                        # Only show if score is above threshold
                        if scores[idx.item()] > 0.4:
                            row = df.iloc[idx.item()]
                            st.markdown(f"**{i+1}. {row['Company Name']}** (Similarity: {scores[idx.item()]:.2f})")
                            st.markdown(f"- 📍 **Address**: {row['Address']}")
                            st.markdown(f"- 🔢 **Consent Number**: {row['Resource Consent Numbers']}")
                            st.markdown(f"- 📜 **Reason**: {row['Reason for Consent']}")
                            st.markdown(f"- ⏳ **Expiry**: {row['Expiry Date'].strftime('%d-%m-%Y') if pd.notna(row['Expiry Date']) else 'Unknown'}")
                            safe_filename = clean_surrogates(row['__file_name__'])
                            st.download_button(
                                label=f"📄 Download PDF: {safe_filename}",
                                data=row['__file_bytes__'],
                                file_name=safe_filename,
                                mime="application/pdf",
                                key=f"download_semantic_{i}"
                            )
                            st.markdown("---")
                elif not keyword_matches_df.empty:
                    st.info("No strong semantic matches, showing keyword-based results:")
                    for i, row in keyword_matches_df.head(5).iterrows():
                        st.markdown(f"**{row['Company Name']}** (Keyword Match)")
                        st.markdown(f"- 📍 **Address**: {row['Address']}")
                        st.markdown(f"- 🔢 **Consent Number**: {row['Resource Consent Numbers']}")
                        st.markdown(f"- 📜 **Reason**: {row['Reason for Consent']}")
                        st.markdown(f"- ⏳ **Expiry**: {row['Expiry Date'].strftime('%d-%m-%Y') if pd.notna(row['Expiry Date']) else 'Unknown'}")
                        safe_filename = clean_surrogates(row['__file_name__'])
                        st.download_button(
                            label=f"📄 Download PDF: {safe_filename}",
                            data=row['__file_bytes__'],
                            file_name=safe_filename,
                            mime="application/pdf",
                            key=f"download_keyword_{i}"
                        )
                        st.markdown("---")
                else:
                    st.warning("No relevant semantic or keyword matches found for your query.")
    else:
        st.warning("No data extracted from the uploaded PDF(s). Please check the file content and ensure they are not just scanned images without OCR capabilities enabled/configured.")
elif not uploaded_files:
    st.info("👈 Please upload PDF files in the sidebar to get started.")

# ----------------------------
# Ask AI About Consents Chatbot (RAG Implemented)
# ----------------------------
st.markdown("---") # Separator
st.markdown("### 💡 Ask AI About Consents")
# Removed the outer expander here to avoid nesting issues
st.markdown("""<div style="background-color:#e0f7fa; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
st.markdown("""
**Ask anything about air discharge consents** (e.g., specific triggers, expiry dates, mitigation measures, or general trends across consents).
The AI will try to answer based on the documents you've uploaded.
""", unsafe_allow_html=True)

llm_provider = st.radio("Choose LLM Provider", ["Gemini", "OpenAI", "Groq"], horizontal=True, key="llm_provider_radio")
chat_input = st.text_area("Enter your question here:", key="chat_input_llm")

if st.button("Ask AI", key="ask_ai_button"):
    if not chat_input.strip():
        st.warning("Please enter a query for the AI.")
    else:
        with st.spinner("AI is thinking..."):
            try:
                # --- RAG: Retrieve relevant documents for LLM context ---
                context_for_llm = []
                if 'df' in locals() and not df.empty:
                    # Re-use the semantic search model for RAG
                    llm_query_norm = normalize_text_for_search(chat_input)
                    llm_query_embedding = model.encode(llm_query_norm, convert_to_tensor=True)

                    # Encode the corpus if not already done, or retrieve from cache
                    corpus_for_llm = (
                        df["Company Name"].fillna("") + " | " +
                        df["Address"].fillna("") + " | " +
                        df["AUP(OP) Triggers"].fillna("") + " | " +
                        df["Mitigation (Consent Conditions)"].fillna("") + " | " +
                        df["Reason for Consent"].fillna("") + " | " +
                        df["Consent Conditions"].fillna("") + " | " +
                        df["Resource Consent Numbers"].fillna("") + " | " +
                        df["Text Blob"].fillna("")
                    ).apply(normalize_text_for_search).tolist()
                    corpus_embeddings_for_llm = model.encode(corpus_for_llm, convert_to_tensor=True)

                    if "e5" in model_name or "bge" in model_name:
                        llm_scores = llm_query_embedding @ corpus_embeddings_for_llm.T
                    else:
                        llm_scores = util.cos_sim(llm_query_embedding, corpus_embeddings_for_llm)[0]

                    # Get top relevant documents for the LLM
                    top_relevant_indices = llm_scores.argsort(descending=True)[:5] # Get top 5
                    
                    # Filter for documents with a reasonable similarity score (e.g., > 0.3)
                    relevant_docs = []
                    for idx in top_relevant_indices:
                        if llm_scores[idx.item()] > 0.3: # Only include documents above a relevance threshold
                            relevant_docs.append(df.iloc[idx.item()])
                    
                    if relevant_docs:
                        # Convert relevant documents to a dictionary format suitable for LLM context
                        context_for_llm = [
                            {
                                "File Name": doc["__file_name__"],
                                "Resource Consent Numbers": doc["Resource Consent Numbers"],
                                "Company Name": doc["Company Name"],
                                "Address": doc["Address"],
                                "Issue Date": doc["Issue Date"].strftime("%Y-%m-%d") if pd.notna(doc["Issue Date"]) else "Unknown",
                                "Expiry Date": doc["Expiry Date"].strftime("%Y-%m-%d") if pd.notna(doc["Expiry Date"]) else "Unknown",
                                "Consent Status": doc["Consent Status Enhanced"],
                                "AUP(OP) Triggers": doc["AUP(OP) Triggers"],
                                "Reason for Consent": doc["Reason for Consent"],
                                "Consent Conditions": doc["Consent Conditions"],
                                "Mitigation (Consent Conditions)": doc["Mitigation (Consent Conditions)"],
                                "Full Document Text Sample": doc["Text Blob"][:1000] + "..." if len(doc["Text Blob"]) > 1000 else doc["Text Blob"] # Provide a snippet
                            }
                            for doc in relevant_docs
                        ]
                    else:
                        st.info("No highly relevant documents found for your query in the uploaded files. Providing general sample data to the AI.")
                        context_for_llm = [{
                            "File Name": "Sample Consent.pdf",
                            "Resource Consent Numbers": "RC12345",
                            "Company Name": "ABC Ltd",
                            "Address": "123 Example St, Auckland",
                            "Issue Date": "2023-01-01",
                            "Expiry Date": "2025-12-31",
                            "Consent Status": "Active",
                            "AUP(OP) Triggers": "E14.1.1",
                            "Reason for Consent": "Discharge of contaminants to air from a boiler.",
                            "Consent Conditions": "1. Emission limits, 2. Monitoring requirements.",
                            "Mitigation (Consent Conditions)": "Dust Management Plan",
                            "Full Document Text Sample": "This is a general sample consent document to provide context."
                        }]
                else:
                    st.info("No documents uploaded. Providing general sample data to the AI.")
                    context_for_llm = [{
                        "File Name": "Sample Consent.pdf",
                        "Resource Consent Numbers": "RC12345",
                        "Company Name": "ABC Ltd",
                        "Address": "123 Example St, Auckland",
                        "Issue Date": "2023-01-01",
                        "Expiry Date": "2025-12-31",
                        "Consent Status": "Active",
                        "AUP(OP) Triggers": "E14.1.1",
                        "Reason for Consent": "Discharge of contaminants to air from a boiler.",
                        "Consent Conditions": "1. Emission limits, 2. Monitoring requirements.",
                        "Mitigation (Consent Conditions)": "Dust Management Plan",
                        "Full Document Text Sample": "This is a general sample consent document to provide context."
                    }]

                # Construct the prompt for the LLM
                ai_system_prompt = """
                You are an environmental compliance assistant specializing in Auckland air discharge consents.
                Your goal is to answer questions accurately and concisely based *only* on the provided consent data.
                Do not make up information. If the answer is not explicitly stated or inferable from the provided data,
                state clearly that you cannot answer based on the given information.
                Prioritize information from the "Full Document Text Sample" and "Consent Conditions" fields if available for details.
                """

                ai_user_query = f"""
                ---
                Relevant Consent Data (JSON array of consent records):
                {context_for_llm}

                ---
                User Query: {chat_input}

                Please provide your answer in a clear, easy-to-read format, using bullet points if appropriate.
                Reference specific consent details (like company name, consent number, or file name) where relevant to support your answer.
                """

                answer_raw = "No response from AI." # Default answer

                if llm_provider == "Gemini":
                    if not google_api_key:
                        st.error("Google API Key is not set. Please set the GOOGLE_API_KEY environment variable.")
                        answer_raw = "Error: Google API Key missing."
                    else:
                        model = genai.GenerativeModel("gemini-pro")
                        response = model.generate_content(ai_user_query)
                        answer_raw = response.text
                elif llm_provider == "OpenAI":
                    if not openai_api_key:
                        st.error("OpenAI API Key is not set. Please set the OPENAI_API_KEY environment variable.")
                        answer_raw = "Error: OpenAI API Key missing."
                    else:
                        messages = [
                            {"role": "system", "content": ai_system_prompt},
                            {"role": "user", "content": ai_user_query}
                        ]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo", # or "gpt-4" if you have access
                            messages=messages,
                            max_tokens=1000, # Increased max_tokens for more detailed answers
                            temperature=0.7
                        )
                        answer_raw = response.choices[0].message.content
                elif llm_provider == "Groq":
                    if not groq_api_key:
                        st.error("Groq API Key is not set. Please set the GROQ_API_KEY environment variable.")
                        answer_raw = "Error: Groq API Key missing."
                    else:
                        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192") # Or "llama3-8b-8192" for faster smaller model
                        groq_response = chat.invoke([
                            {"role": "system", "content": ai_system_prompt},
                            {"role": "user", "content": ai_user_query}
                        ])
                        answer_raw = groq_response.content if hasattr(groq_response, 'content') else str(groq_response)

                st.markdown(f"### 🧠 Answer from {llm_provider} AI\n\n{answer_raw}")
                # Log the chat after a successful response
                log_ai_chat(chat_input, answer_raw)

            except Exception as e:
                st.error(f"An error occurred with the AI request: {e}")
                st.info("Please check your API keys and internet connection, or try a different query.")
st.markdown("</div>", unsafe_allow_html=True) # Closes the inner div for the chatbot input area

# ----------------------------
# AI Chat Log (Now a separate top-level expander)
# ----------------------------
st.markdown("---") # Separator
st.markdown("### 💬 AI Chat Log")
with st.expander("View AI Chat Log", expanded=False): # This expander is now at the top level
    csv_log = get_chat_log_as_csv()
    if csv_log:
        st.download_button(
            label="Download Chat Log as CSV",
            data=csv_log,
            file_name="ai_chat_log.csv",
            mime="text/csv",
            key="download_chat_log"
        )
        df_chat_log = pd.read_csv(io.StringIO(csv_log.decode('utf-8')))
        st.dataframe(df_chat_log)
    else:
        st.info("No chat history available yet.")

# --------------------
# Footer
# --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #656e6b; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True)
