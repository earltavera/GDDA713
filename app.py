# Auckland Air Discharge Consent Dashboard - Complete with Gemini, OpenAI, and Groq Chatbot

import streamlit as st
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

import pandas as pd
import pymupdf
fitz = pymupdf
import regex as re
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
from openai import OpenAI
import google.generativeai as genai
from langchain_groq import ChatGroq

# Load Environment Variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = OpenAI()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Weather Function
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

# Banner
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
# Utility Functions
# --------------------

def check_expiry(expiry_date):
    if expiry_date is None or pd.isna(expiry_date):
        return "Unknown"
    # Ensure we are comparing datetime objects
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
    # This function remains the same as before
    # RC number patterns
    rc_raw = [
        r"Application number:\s*(.+?)(?=\s*Applicant)",
        r"Application numbers:\s*(.+)(?=\s*Applicant)",
        r"Application number\(s\):\s*(.+)(?=\s*Applicant)",
        r"RC[0-9]{5,}"
    ]
    rc_matches = []
    for pattern in rc_raw:
        rc_matches += re.findall(pattern, text, re.IGNORECASE)
    rc_str = "".join(dict.fromkeys(rc_matches))

    # Company name patterns
    company_raw = [
        r"Applicant:\s*(.+?)(?=\s*Site address)",
        r"Applicant's name:\s*(.+?)(?=\s*Site address)"
    ]
    company_matches = []
    for pattern in company_raw:
        company_matches += re.findall(pattern, text)
    company_str = "".join(dict.fromkeys(company_matches))

    # Address pattern
    address_raw = r"Site address:\s*(.+?)(?=\s*Legal description)"
    address_matches = re.findall(address_raw, text)
    address_str = "".join(dict.fromkeys(address_matches))

    # Issue date patterns
    issue_date_raw = [
        r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})",
        r"Date:\s*(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\b\d{1,2} [A-Za-z]+ \d{4}\b)",
        r"Date:\s*(\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)"
    ]
    issue_matches = []
    for pattern in issue_date_raw:
        issue_matches += re.findall(pattern, text)
    issue_str = "".join(dict.fromkeys(issue_matches))
    issue_date = parse_mixed_date(issue_str)

    # Expiry date patterns
    expiry_raw = [
        r"expire on (\d{1,2} [A-Za-z]+ \d{4})",
        r"expires on (\d{1,2} [A-Za-z]+ \d{4})",
        r"expires (\d{1,2} [A-Za-z]+ \d{4})",
        r"expire (\d{1,2} [A-Za-z]+\d{4})",
        r"(\d{1,} years) from the date of commencement",
        r"DIS\d{5,}(?:-w+)?\b will expire (\d{1,} years [A-Za-z]+[.?!])",
        r"expires (\d{1,} months [A-Za-z])+[.?!]",
        r"expires on (\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b)",
        r"expire on (\d{1,2}/\d{1,2}/\d{4})",
        r"expire ([A-Za-z](\d{1,}) years)",
        r"expires (\d{1,} years [A-Za-z]+[.?1])"
    ]
    expiry_matches = []
    for pattern in expiry_raw:
        expiry_matches += re.findall(pattern, text)
    expiry_str = "".join(dict.fromkeys(expiry_matches))
    expiry_date = parse_mixed_date(expiry_str)

    # AUP triggers
    trigger_raw = [
        r"(E14\.\d+\.\d+)",
        r"(E14\.\d+\.)",
        r"(NES:STO)",
        r"(NES:AQ)"
    ]
    trigger_matches = []
    for pattern in trigger_raw:
        trigger_matches += re.findall(pattern, text)
    triggers_str = " ".join(dict.fromkeys(trigger_matches))

    # Proposal
    proposal_raw = r"Proposal\s*:\s*(.+?)(?=\n[A-Z]|\.)"
    proposal_matches = re.findall(proposal_raw, text, re.DOTALL)
    proposal_str = " ".join(proposal_matches)

    # Consent Conditions
    conditions_raw = [
        r"(?<=Conditions).*?(?=Advice notes)", r"(?<=Specific conditions - Air Discharge DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions -)",
        r"(?<=Air Quality conditions).*?(?=Wastewater Discharge conditions)", r"(?<=Air Discharge Permit Conditions).*?(?=E. Definitions)",
        r"(?<=Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=DIS\d{5,}(?:-\w+)?\b)", r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?<=Conditions Specific to air quality).*?(?=Advice notes)", r"(?<=Specific conditions - air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=regional discharge DIS\d{5,}(?:-w+)?\b).*?(?=Advice notes)", r"(?<=Specific conditions - discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Consolidated conditions of consent as amended).*?(?=Advice notes)", r"(?<=Specific conditions - Air Discharge DIS\d{5,}\b).*?(?=Advice notes)",
        r"(?<=Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=Advice notes)",
        r"(?<=DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=DIS\d{5,}(?:-\w+)?\b - Specific conditions)", r"(?<=Specific Conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?<=Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b Only).*?(?=Advice notes)", r"(?<=Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Specific Conditions -)",
        r"(?<=SPECIFIC CONDITIONS - DISCHARGE TO AIR DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=Conditions relevant to Discharge Permit DIS\d{5,}(?:-\w+)?\b only).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=Specific conditions - air discharge permit (DIS\d{5,}(?:-\w+)?\b)).*?(?=Advice notes)",
        r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b (air)).*?(?=Advice notes)", r"(?<=Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specifc conditions)",
        r"(?<=Attachment 1: Consolidated conditions of consent as amended).*?(?=Advice notes)", r"(?<=Specific Air Discharge Conditions).*?(?=Advice notes)",
        r"(?<=Specific conditions - Discharge to Air: DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=Specific conditions - discharge permit (air discharge) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Air Discharge Limits).*?(?= Acoustic Conditions)", r"(?<=Specific conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit (s15) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Secific conditions)",
        r"(?<=Specific conditions relating to Air discharge permit - DIS\d{5,}(?:-\w+)?\b).*?(?=General Advice notes)", r"(?<=Specific conditions - Discharge permit (s15) - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific Conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)", r"(?<=Specific conditions - Discharge to air: DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?<=Attachement 1: Consolidated conditions of consent as amended).*?(?=Resource Consent Notice of Works Starting)", r"(?<=Specific conditions - Air Discharge consent - DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?<=Specific conditions - Discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)", r"(?<=DIS\d{5,}(?:-\w+)?\b - Air Discharge).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?<=DIS\d{5,}(?:-\w+)?\b & DIS\d{5,}(?:-\w+)?\b).*?(?=).*?(?=SUB\d{5,}\b) - Subdivision", r"(?<=Specific conditions - Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice Notes - General)",
        r"(?<=AIR QUALITY - ROCK CRUSHER).*?(?=GROUNDWATER)"
    ]
    conditions_matches = []
    for pattern in conditions_raw:
        conditions_matches += re.findall(pattern, text, re.DOTALL)
    conditions_str = " ".join(conditions_matches)
    conditions_numbers = re.findall(r"^\d+(?=\.)", conditions_str, re.MULTILINE)

    management_plan = re.findall(r"(?i)\b(\w+)\sManagement Plan", conditions_str)
    managementplan_final = list(dict.fromkeys([f"{word} Management Plan" for word in management_plan]))

    return {
        "Resource Consent Numbers": rc_str, "Company Name": company_str, "Address": address_str,
        "Issue Date": issue_date.strftime("%d-%m-%Y") if issue_date else "Unknown",
        "Expiry Date": expiry_date.strftime("%d-%m-%Y") if expiry_date else "Unknown",
        "AUP(OP) Triggers": triggers_str, "Reason for Consent": proposal_str,
        "Consent Conditions": ", ".join(conditions_numbers), "Mitigation (Consent Conditions)": ", ".join(managementplan_final),
        "Consent Status": check_expiry(expiry_date) if 'expiry_date' in locals() and expiry_date else "Unknown",
        "Text Blob": text
    }


def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer_raw):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer_raw}
    file_path = "ai_chat_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Timestamp", "Question", "Answer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# --- All other functions like get_chat_log_as_csv remain the same ---

# --------------------
# Sidebar & Model Loader
# --------------------

st.sidebar.markdown("""
    <h2 style='color:#2c6e91; font-family:Segoe UI, Roboto, sans-serif;'>
        Control Panel
    </h2>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("Choose LLM model:", [
    "all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"
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
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

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
        # Convert date strings to datetime objects
        df["Issue Date"] = pd.to_datetime(df["Issue Date"], format='%d-%m-%Y', errors='coerce')
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], format='%d-%m-%Y', errors='coerce')
        
        df["GeoKey"] = df["Address"].str.lower().str.strip()
        lat_lon = df["GeoKey"].apply(geocode_address)
        df["Latitude"], df["Longitude"] = zip(*lat_lon)

        # Apply check_expiry after date conversion
        df["Consent Status"] = df["Expiry Date"].apply(check_expiry)
        
        df["Consent Status Enhanced"] = df["Consent Status"]
        ninety_days = datetime.now() + timedelta(days=90)
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"] <= ninety_days),
            "Consent Status Enhanced"
        ] = "Expiring in 90 Days"
        st.session_state.df = df

# Display dashboard elements if DataFrame exists in session state
if not st.session_state.df.empty:
    df = st.session_state.df
    # --- This entire dashboard section remains the same. It will now use the processed df. ---
    # Metrics
    st.subheader("Consent Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Consents", len(df))
    col2.metric("Expired", df["Consent Status"].value_counts().get("Expired", 0))
    col3.metric("Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum())

    # Status Chart
    status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
    status_counts.columns = ["Consent Status", "Count"]
    color_map = {"Unknown": "gray", "Expired": "red", "Active": "green", "Expiring in 90 Days": "orange"}
    fig_status = px.bar(status_counts, x="Consent Status", y="Count", text="Count", color="Consent Status", color_discrete_map=color_map)
    fig_status.update_traces(textposition="outside")
    fig_status.update_layout(title="Consent Status Overview", title_x=0.5)
    st.plotly_chart(fig_status, use_container_width=True)

    # Consent Table and Map... (code is identical, so omitted for brevity but should be kept in your file)
    with st.expander("Consent Table", expanded=True):
        # ... same code as before ...
        pass # Placeholder for your existing identical code
    with st.expander("Consent Map", expanded=True):
        # ... same code as before ...
        pass # Placeholder for your existing identical code
    with st.expander("Semantic Search Results", expanded=True):
        # ... same code as before ...
        pass # Placeholder for your existing identical code


# ----------------------------
# Ask AI About Consents Chatbot (UPGRADED LOGIC)
# ----------------------------
st.markdown("### 🤖 Ask AI About Consents")
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#d1eaf0; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask the AI to analyze all uploaded consent data.** (e.g., 'Which consents are for quarries?' or 'List all consents that expire in 2028')", unsafe_allow_html=True)

    llm_provider = st.radio("Choose LLM Provider", ["Groq", "Gemini", "OpenAI"], horizontal=True)
    chat_input = st.text_area("Your question for the AI:", key="chat_input")

    if st.button("Ask AI"):
        df = st.session_state.get('df', pd.DataFrame()) # Safely get the dataframe
        if not chat_input.strip():
            st.warning("Please enter a query.")
        elif df.empty:
            st.warning("Please upload and process PDF files before asking the AI.")
        else:
            with st.spinner(f"Asking {llm_provider} AI to analyze all {len(df)} consents..."):
                try:
                    # Create a comprehensive context from the entire DataFrame
                    context_df = df[[
                        "Company Name", "Address", "Resource Consent Numbers",
                        "Consent Status Enhanced", "Expiry Date", "Reason for Consent",
                        "Mitigation (Consent Conditions)", "AUP(OP) Triggers"
                    ]].copy()
                    context_df['Expiry Date'] = context_df['Expiry Date'].dt.strftime('%Y-%m-%d').fillna('N/A')
                    
                    full_context_csv = context_df.to_csv(index=False)

                    # Safeguard for very large context windows
                    # Approx. 4 chars per token, Llama3-70B is 8k tokens. 8000*4 = 32000 chars.
                    if len(full_context_csv) > 30000:
                        sample_size = int(len(df) * (30000 / len(full_context_csv)))
                        st.warning(f"The data from {len(df)} consents is too large for the AI's context window. Analyzing a sample of {sample_size} consents instead.")
                        full_context_csv = context_df.sample(n=sample_size, random_state=1).to_csv(index=False)

                    # --- New Unified Prompting Strategy ---
                    system_prompt = """You are an expert AI data analyst for Auckland Council resource consents.
                    Your task is to answer the user's query based *only* on the data provided in the user's message.
                    The data will be in CSV format. Do not use any external knowledge.
                    If the answer cannot be found in the provided data, state that clearly.
                    When referring to a specific consent, mention the company name or consent number.
                    Present your answer in clear, easy-to-read markdown."""

                    user_prompt = f"""
                    --- CONSENT DATA (CSV format) ---
                    {full_context_csv}
                    --- END OF DATA ---

                    Based on the data above, please answer the following query: "{chat_input}"
                    """
                    
                    answer_raw = ""
                    if llm_provider == "Groq":
                        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                        groq_response = chat.invoke(messages)
                        answer_raw = groq_response.content
                    
                    elif llm_provider == "Gemini":
                        model_genai = genai.GenerativeModel("gemini-pro")
                        # Gemini works best with a single combined prompt
                        response = model_genai.generate_content(system_prompt + "\n" + user_prompt)
                        answer_raw = response.text

                    elif llm_provider == "OpenAI":
                        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo", messages=messages, max_tokens=1000, temperature=0.5
                        )
                        answer_raw = response.choices[0].message.content

                    st.markdown(f"### 🧠 Answer from {llm_provider} AI")
                    st.markdown(answer_raw)
                    log_ai_chat(chat_input, answer_raw)

                except Exception as e:
                    st.error(f"AI error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True)
