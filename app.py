import streamlit as st
import pandas as pd
import fitz  # Simplified import
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
import pytz
import json

# --- LLM Specific Imports ---
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# --- API Key Setup ---
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
openweathermap_api_key = os.getenv("OPENWEATHER_API_KEY") or st.secrets.get("OPENWEATHER_API_KEY")

# ------------------------
# Streamlit Page Config & Style (MUST BE THE FIRST STREAMLIT COMMAND)
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    st.error("Google API key not found. Gemini AI features will be offline.")

# --- Weather Function ---
@st.cache_data(ttl=600)
def get_auckland_weather():
    if not openweathermap_api_key:
        return "Sunny, 18°C (offline mode)"
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
        return pd.NaT

    auckland_tz = pytz.timezone("Pacific/Auckland")
    if dt.tzinfo is None:
        try:
            return auckland_tz.localize(dt, is_dst=None)
        except (pytz.AmbiguousTimeError, pytz.NonExistentTimeError):
            return pd.NaT # Or a more robust fallback
    else:
        return dt.astimezone(auckland_tz)

def check_expiry(expiry_date):
    """
    REFINED: Checks the status of a consent based on a pre-localized timezone-aware expiry date.
    """
    if pd.isna(expiry_date):
        return "Unknown"
    
    current_nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
    return "Expired" if expiry_date < current_nz_time else "Active"


@st.cache_data(show_spinner=False)
def geocode_address(address):
    if not isinstance(address, str):
        return (None, None)
    standardized_address = address.strip()
    if not re.search(r'auckland', standardized_address, re.IGNORECASE):
        standardized_address += ", Auckland"
    if not re.search(r'new zealand|nz', standardized_address, re.IGNORECASE):
        standardized_address += ", New Zealand"

    try:
        geolocator = Nominatim(user_agent="auckland_air_discharge_dashboard_v2")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        location = geocode(standardized_address)
        if location:
            return (location.latitude, location.longitude)
        else:
            return (None, None)
    except Exception as e:
        st.warning(f"Geocoding failed for '{standardized_address}': {e}")
        return (None, None)

@st.cache_data(show_spinner="Extracting metadata with AI...")
def extract_metadata_with_llm(text_blob: str, file_name: str) -> dict:
    """
    IMPLEMENTED: Extracts metadata from PDF text using a generative AI model.
    """
    if not google_api_key:
        st.warning("Cannot use AI extraction: Google API Key is missing. Will attempt regex fallback.")
        return {}

    try:
        model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    except Exception as e:
        st.error(f"Failed to initialize Gemini model for extraction: {e}")
        return {}

    json_schema = {
        "Resource Consent Numbers": "string (e.g., 'DIS12345, SUB67890')",
        "Company Name": "string",
        "Address": "string (full site address)",
        "Issue Date": "string (format as DD-MM-YYYY)",
        "Expiry Date": "string (format as DD-MM-YYYY, or a duration like '35 years')",
        "AUP(OP) Triggers": "string (e.g., 'E14.1.1, NES:AQ')",
        "Reason for Consent": "string (a brief summary of the proposal)",
        "Consent Conditions": "string (the full text of all specific air discharge conditions)"
    }
    prompt = f"""
    Analyze the following text from a resource consent document named '{file_name}'. Your task is to extract the specified information accurately and format it as a JSON object.

    Instructions:
    1.  Read the entire text carefully to understand its structure.
    2.  Extract the information based on the keys provided in the desired JSON schema.
    3.  For dates, if a specific date (e.g., "31 October 2035") is found, format it as DD-MM-YYYY.
    4.  If an expiry is given as a duration (e.g., "thirty five years from the date of commencement"), extract the duration text verbatim (e.g., "thirty five years").
    5.  If a piece of information is not found, use "Unknown".
    6.  The "Consent Conditions" should be a consolidated block of text containing all numbered or lettered conditions related specifically to the air discharge permit.
    7.  Your output MUST be ONLY the JSON object, without any surrounding text or markdown formatting.

    Desired JSON Schema:
    {json.dumps(json_schema, indent=2)}

    --- DOCUMENT TEXT START ---
    {text_blob[:250000]}
    --- DOCUMENT TEXT END ---

    JSON Output:
    """
    try:
        response = model.generate_content(prompt)
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "")
        extracted_data = json.loads(cleaned_response_text)
        extracted_data["Text Blob"] = text_blob
        return extracted_data
    except json.JSONDecodeError:
        st.warning(f"AI returned invalid JSON for {file_name}. Attempting regex fallback.")
        return {}
    except Exception as e:
        st.warning(f"AI extraction error for {file_name}: {e}. Attempting regex fallback.")
        return {}

def extract_metadata(text):
    """
    The original regex function, now serving as a fallback. Patterns have been consolidated.
    """
    # Consolidated RC number patterns
    rc_patterns = [
        r"Application number(?:s|s)?:?\s*(.+?)(?=\s*Applicant|\s*Original consent)",
        r"RC[0-9]{5,}"
    ]
    rc_matches = []
    for pattern in rc_patterns:
        rc_matches.extend(re.findall(pattern, text, re.IGNORECASE | re.DOTALL))
    rc_str = ", ".join(list(dict.fromkeys(m.strip() for m in rc_matches if m)))

    # Consolidated Company name patterns
    company_pattern = r"Applicant(?:'s name)?:\s*(.+?)(?=\s*Site address)"
    company_matches = re.findall(company_pattern, text, re.IGNORECASE)
    company_str = ", ".join(list(dict.fromkeys(m.strip() for m in company_matches if m)))

    # Address pattern
    address_pattern = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_match = re.findall(address_pattern, text, re.MULTILINE | re.IGNORECASE)
    address_str = ", ".join(list(dict.fromkeys(m.strip() for m in address_match if m)))

    # Issue date patterns
    issue_date_patterns = [
        r"Commissioner\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}[ /][A-Za-z]+[ /]\d{2,4})",
        r"Date:\s*(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)"
    ]
    issue_date = pd.NaT
    for pattern in issue_date_patterns:
        matches = re.findall(pattern, text)
        if matches:
            date_str = matches[0]
            issue_date = pd.to_datetime(date_str, errors='coerce', dayfirst=True)
            if pd.notna(issue_date):
                break
    
    # Consolidated Expiry patterns
    expiry_patterns = [
        r"expires?\s+(?:on\s+)?the\s+(\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?\s*[A-Za-z]+\s+\d{4})",
        r"expires?\s+(?:on\s+)?(\d{1,2}[ /][A-Za-z]+[ /]\d{4})",
        r"expires?\s+(\d{1,2}\s+years)",
        r"expire\s+([a-zA-Z]+\s+years)" # e.g. "five years"
    ]
    expiry_date_val = "Unknown Expiry Date"
    for pattern in expiry_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            expiry_date_val = matches[0]
            break

    # AUP triggers
    triggers_str = " ".join(list(dict.fromkeys(re.findall(r"(E14\.\d+\.\d+|NES:STO|NES:AQ)", text))))

    # Reason (Proposal)
    proposal_pattern = r"Proposal|Introduction and summary of proposal|Summary of Decision\s*(.+?)(?=\n[A-Z]|\.|\:)"
    proposal_match = re.search(proposal_pattern, text, re.DOTALL)
    proposal_str = proposal_match.group(1).strip() if proposal_match else "Unknown Reason for Consent"
    
    # Conditions
    conditions_str = ""
    conditions_pattern = r"(?<=Conditions).*?(?=Advice notes)"
    conditions_match = re.search(conditions_pattern, text, re.DOTALL | re.IGNORECASE)
    if conditions_match:
        conditions_str = conditions_match.group(0).strip()

    return {
        "Resource Consent Numbers": rc_str or "Unknown",
        "Company Name": company_str or "Unknown",
        "Address": address_str or "Unknown",
        "Issue Date": issue_date, # Return datetime object or NaT
        "Expiry Date": expiry_date_val, # Return string (date or duration)
        "AUP(OP) Triggers": triggers_str or "Unknown",
        "Reason for Consent": proposal_str,
        "Consent Conditions": conditions_str or "Unknown",
        "Text Blob": text
    }

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

# --- LOGGING FUNCTIONS (Unchanged) ---
def log_ai_chat(question, answer):
    #... (Your existing function is good)
    pass
def get_chat_log_as_csv():
    #... (Your existing function is good)
    pass


# --- Sidebar & Model Loader ---
st.sidebar.markdown("## Control Panel")
model_name = st.sidebar.selectbox("Choose Embedding Model:", ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"])
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query")

@st.cache_resource
def load_embedding_model(name):
    return SentenceTransformer(name)

embedding_model = load_embedding_model(model_name)

@st.cache_data(show_spinner="Generating document embeddings...")
def get_corpus_embeddings(text_blobs_tuple, model_name_str):
    model_obj = load_embedding_model(model_name_str)
    return model_obj.encode(list(text_blobs_tuple), convert_to_tensor=True)

# --- File Processing & Dashboard ---
if uploaded_files:
    all_data = []
    with st.spinner("Processing uploaded PDF files..."):
        for file in uploaded_files:
            try:
                file_bytes = file.read()
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "\n".join(page.get_text() for page in doc)
                
                # IMPLEMENTED: Use LLM extraction with regex fallback
                data = extract_metadata_with_llm(text, file.name)
                if not data or not data.get("Company Name"):
                    data = extract_metadata(text) # Fallback call
                
                if data:
                    data["__file_name__"] = file.name
                    data["__file_bytes__"] = file_bytes
                    all_data.append(data)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # --- Enhanced Date Processing ---
        df['Issue Date'] = pd.to_datetime(df['Issue Date'], format='%d-%m-%Y', errors='coerce')

        def calculate_expiry(row):
            expiry_info = row['Expiry Date']
            issue_date = row['Issue Date']
            if pd.isna(expiry_info):
                return pd.NaT
            if isinstance(expiry_info, str):
                # Check for relative years (e.g., "35 years")
                year_match = re.search(r'(\d+)\s+years', expiry_info, re.IGNORECASE)
                if year_match and pd.notna(issue_date):
                    years_to_add = int(year_match.group(1))
                    try:
                        return issue_date.replace(year=issue_date.year + years_to_add)
                    except ValueError: # handles leap years issues
                        return issue_date + timedelta(days=365.25 * years_to_add)
                # Try parsing as a regular date
                return pd.to_datetime(expiry_info, dayfirst=True, errors='coerce')
            return pd.to_datetime(expiry_info, errors='coerce')

        df['Expiry Date Calculated'] = df.apply(calculate_expiry, axis=1)
        
        # Localize dates
        df['Issue Date'] = df['Issue Date'].apply(localize_to_auckland)
        df['Expiry Date'] = df['Expiry Date Calculated'].apply(localize_to_auckland)
        
        # Geocoding
        df["GeoKey"] = df["Address"].str.lower().str.strip()
        df["Latitude"], df["Longitude"] = zip(*df["GeoKey"].apply(geocode_address))

        # Status Calculation using simplified function
        df["Consent Status"] = df['Expiry Date'].apply(check_expiry)
        df["Consent Status Enhanced"] = df["Consent Status"]

        current_nz_aware_time = datetime.now(pytz.timezone("Pacific/Auckland"))
        ninety_days_from_now = current_nz_aware_time + timedelta(days=90)

        expiring_mask = (df["Consent Status"] == "Active") & \
                        (df["Expiry Date"] > current_nz_aware_time) & \
                        (df["Expiry Date"] <= ninety_days_from_now)
        df.loc[expiring_mask, "Consent Status Enhanced"] = "Expiring in 90 Days"
        
        # --- Dashboard Metrics (Refactored to use st.metric) ---
        st.subheader("Consent Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)

        total_consents = len(df)
        expiring_90_days = (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum()
        expired_count = (df["Consent Status"] == "Expired").sum()
        active_count = (df["Consent Status Enhanced"] == "Active").sum()

        col1.metric("Total Consents", total_consents)
        col2.metric("Expiring in 90 Days", expiring_90_days)
        col3.metric("Total Expired", expired_count)
        col4.metric("Currently Active", active_count)


        # --- Charts and Tables (Largely unchanged, but using refined data) ---
        color_map = {"Unknown": "gray", "Expired": "#d9534f", "Active": "#5cb85c", "Expiring in 90 Days": "#f0ad4e"}
        status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
        status_counts.columns = ["Consent Status", "Count"]
        fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status",
                            color_discrete_map=color_map, title="Consent Status Overview")
        st.plotly_chart(fig_status, use_container_width=True)

        with st.expander("Consent Table", expanded=True):
            status_filter = st.selectbox("Filter by Status", ["All"] + df["Consent Status Enhanced"].unique().tolist())
            filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
            
            # Formatting dates for display
            display_df = filtered_df.copy()
            display_df['Issue Date'] = display_df['Issue Date'].dt.strftime('%d-%b-%Y')
            display_df['Expiry Date'] = display_df['Expiry Date'].dt.strftime('%d-%b-%Y')

            st.dataframe(display_df[[
                "__file_name__", "Resource Consent Numbers", "Company Name", "Address", "Issue Date", "Expiry Date",
                "Consent Status Enhanced", "AUP(OP) Triggers", "Reason for Consent"
            ]].rename(columns={"__file_name__": "File Name", "Consent Status Enhanced": "Consent Status"}))
            # ... (Download button code remains the same)

        with st.expander("Consent Map", expanded=True):
            map_df = df.dropna(subset=["Latitude", "Longitude"])
            if not map_df.empty:
                 map_df_display = map_df.copy()
                 map_df_display['Expiry Date Str'] = map_df_display['Expiry Date'].dt.strftime('%d-%b-%Y')
                 fig = px.scatter_mapbox(map_df_display, lat="Latitude", lon="Longitude",
                                        hover_name="Company Name",
                                        hover_data={"Address": True, "Consent Status Enhanced": True, "Expiry Date Str": True},
                                        zoom=9.5, color="Consent Status Enhanced", color_discrete_map=color_map)
                 fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                 st.plotly_chart(fig, use_container_width=True)

        # ... (Semantic Search and AI Chatbot sections remain the same as they were well-structured) ...
        # Semantic Search
        with st.expander("Semantic Search Results", expanded=True):
             if query_input and not df.empty:
                corpus = df["Text Blob"].tolist()
                corpus_embeddings = get_corpus_embeddings(tuple(corpus), model_name)
                
                query_embedding = embedding_model.encode(query_input, convert_to_tensor=True)
                scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                top_k_indices = scores.argsort(descending=True)

                similarity_threshold = st.slider("Semantic Search Relevance Threshold", 0.0, 1.0, 0.5, 0.05)
                
                results_found = 0
                for idx in top_k_indices[:10]: # Limit to top 10 potential results
                    score = scores[idx.item()]
                    if score >= similarity_threshold:
                        results_found += 1
                        row = df.iloc[idx.item()]
                        st.markdown(f"**{results_found}. {row['Company Name']} - {row['Address']}** (Relevance: {score:.2f})")
                        expiry_display = row['Expiry Date'].strftime('%d-%b-%Y') if pd.notna(row['Expiry Date']) else 'N/A'
                        st.markdown(f"- **Status**: {row['Consent Status Enhanced']} (Expires: {expiry_display})")
                        st.markdown(f"- **Triggers**: {row['AUP(OP) Triggers']}")
                        
                        safe_filename = clean_surrogates(row['__file_name__'])
                        st.download_button(label=f"Download PDF ({safe_filename})", data=row['__file_bytes__'], file_name=safe_filename, mime="application/pdf", key=f"download_{idx.item()}")
                        st.markdown("---")
                
                if results_found == 0:
                    st.info(f"No documents found above the {similarity_threshold:.2f} relevance threshold.")


# --- ASK AI CHATBOT (Unchanged) ---
# ... Your existing chatbot code is solid and can be pasted here directly ...
st.markdown("---")
st.subheader("Ask AI About Consents")

with st.expander("AI Chatbot", expanded=True):
    # The rest of your chatbot code is good and can be placed here.
    pass # Placeholder for your existing chatbot UI and logic

st.markdown("---")
st.caption("Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025")
