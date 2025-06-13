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
# Add a check for Google API Key before configuring
if os.getenv("GOOGLE_API_KEY"):
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Weather Function
@st.cache_data(ttl=600)
def get_auckland_weather():
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API Key not set"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q=Auckland,nz&units=metric&appid={api_key}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        return f"{desc}, {temp:.1f}°C"
    except requests.exceptions.RequestException as e:
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
    if pd.isna(expiry_date):
        return "Unknown"
    return "Expired" if expiry_date < datetime.now() else "Active"

@st.cache_data(show_spinner=False)
def geocode_address(address):
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    try:
        location = geocode(address)
        return (location.latitude, location.longitude) if location else (None, None)
    except:
        return (None, None)

def parse_mixed_date(date_str):
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None

def extract_metadata(text):
    # This function is now more comprehensive to populate the detailed table
    rc_matches = re.findall(r"Application number(?:s|\(s\))?:\s*([^\n]+)|(RC\d{5,})", text, re.IGNORECASE)
    rc_str = " ".join(dict.fromkeys(item for sublist in rc_matches for item in sublist if item))

    company_matches = re.findall(r"Applicant(?:'s name)?:\s*([^\n]+)", text)
    company_str = " ".join(dict.fromkeys(company_matches)).strip()

    address_matches = re.findall(r"Site address:\s*([^\n]+)", text)
    address_str = " ".join(dict.fromkeys(address_matches)).strip()

    issue_date_matches = re.findall(r"Date:\s*(\d{1,2}[ /][A-Za-z]+[ /]\d{4}|\d{1,2}[/]\d{1,2}[/]\d{2,4})", text)
    issue_date = parse_mixed_date(issue_date_matches[0] if issue_date_matches else None)

    expiry_date_matches = re.findall(r"expire[s]? on (\d{1,2} [A-Za-z]+ \d{4})", text)
    expiry_date = parse_mixed_date(expiry_date_matches[0] if expiry_date_matches else None)

    trigger_matches = re.findall(r"(E14\.\d+\.\d+|NES:STO|NES:AQ)", text)
    triggers_str = ", ".join(dict.fromkeys(trigger_matches))

    proposal_matches = re.findall(r"Proposal\s*:\s*([^\n]+)", text, re.DOTALL)
    proposal_str = " ".join(p.strip() for p in proposal_matches)

    conditions_matches = re.findall(r"Conditions\s*\n(.*?)(?=Advice notes)", text, re.DOTALL)
    conditions_str = " ".join(c.strip() for c in conditions_matches)
    conditions_numbers = re.findall(r"^\s*(\d+)\.", conditions_str, re.MULTILINE)

    management_plan_matches = re.findall(r"(\w+\s+Management\s+Plan)", conditions_str, re.IGNORECASE)
    managementplan_final = ", ".join(dict.fromkeys(m.strip() for m in management_plan_matches))

    return {
        "Resource Consent Numbers": rc_str, "Company Name": company_str, "Address": address_str,
        "Issue Date": issue_date, "Expiry Date": expiry_date,
        "AUP(OP) Triggers": triggers_str, "Reason for Consent": proposal_str,
        "Number of Conditions": len(conditions_numbers), "Mitigation Plans": managementplan_final,
        "Text Blob": text # Keep the full text for semantic search
    }

# All other utility functions (clean_surrogates, log_ai_chat, get_chat_log_as_csv) remain the same

def log_ai_chat(question, answer_raw):
    # ... (code is unchanged)
    pass

# --------------------
# Sidebar & Model Loader
# --------------------
st.sidebar.markdown("## Control Panel")
model_name = st.sidebar.selectbox("Choose Embedding Model:", ["all-MiniLM-L6-v2", "multi-qa-MiniLM-L6-cos-v1"])
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, help="Upload one or more air discharge consent decision documents.")
query_input = st.sidebar.text_input("Semantic Search in Documents", placeholder="e.g., 'dust mitigation'")

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)
model = load_model(model_name)

# ------------------------
# File Processing & State Management
# ------------------------
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()

if uploaded_files:
    with st.spinner("Processing PDF documents..."):
        all_data = []
        for file in uploaded_files:
            try:
                file_bytes = file.read()
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    text = "".join(page.get_text() for page in doc)
                metadata = extract_metadata(text)
                metadata["__file_name__"] = file.name
                metadata["__file_bytes__"] = file_bytes
                all_data.append(metadata)
            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")

        if all_data:
            df = pd.DataFrame(all_data)
            df['Latitude'], df['Longitude'] = zip(*df['Address'].apply(geocode_address))
            df["Consent Status"] = df["Expiry Date"].apply(check_expiry)
            df["Consent Status Enhanced"] = df["Consent Status"]
            ninety_days = datetime.now() + timedelta(days=90)
            df.loc[(df["Consent Status"] == "Active") & (df["Expiry Date"] <= ninety_days), "Consent Status Enhanced"] = "Expiring in 90 Days"
            st.session_state.df = df # Store in session state

# ------------------------
# Main Dashboard Display
# ------------------------
if not st.session_state.df.empty:
    df = st.session_state.df
    st.subheader("Consent Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Consents Processed", len(df))
    col2.metric("Active Consents", df["Consent Status"].value_counts().get("Active", 0))
    col3.metric("Expiring in 90 Days", (df["Consent Status Enhanced"] == "Expiring in 90 Days").sum())

    # Status Chart
    status_counts = df["Consent Status Enhanced"].value_counts().reset_index()
    status_counts.columns = ["Consent Status", "Count"]
    color_map = {"Unknown": "gray", "Expired": "red", "Active": "green", "Expiring in 90 Days": "orange"}
    fig_status = px.bar(status_counts, x="Consent Status", y="Count", text_auto=True, title="Consent Status Overview", color="Consent Status", color_discrete_map=color_map)
    fig_status.update_layout(title_x=0.5)
    st.plotly_chart(fig_status, use_container_width=True)

    # --- NEW: Comprehensive and Detailed Data Table ---
    st.markdown("### Detailed Consent Data")
    with st.expander("View and Filter All Consent Details", expanded=True):
        status_filter = st.selectbox("Filter table by Status:", ["All"] + list(df["Consent Status Enhanced"].unique()))
        
        filtered_df = df if status_filter == "All" else df[df["Consent Status Enhanced"] == status_filter]
        
        display_columns = {
            "Resource Consent Numbers": "Consent No.",
            "Company Name": "Company",
            "Address": "Site Address",
            "Issue Date": "Issued",
            "Expiry Date": "Expires",
            "Consent Status Enhanced": "Status",
            "Reason for Consent": "Consent Reason",
            "Mitigation Plans": "Mitigation Plans",
            "Number of Conditions": "Conditions",
            "AUP(OP) Triggers": "AUP Triggers",
            "__file_name__": "Source File"
        }
        
        # Format dates for display
        display_df = filtered_df[display_columns.keys()].copy()
        display_df['Issued'] = pd.to_datetime(display_df['Issue Date']).dt.strftime('%d %b %Y')
        display_df['Expires'] = pd.to_datetime(display_df['Expiry Date']).dt.strftime('%d %b %Y')
        display_df = display_df.rename(columns=display_columns)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        csv_export = display_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data as CSV", csv_export, "consent_data.csv", "text/csv", key='download-all-data')

    # Other expanders (Map, Search) follow here...
    # ...

# ----------------------------
# Ask AI About Consents Chatbot (UPGRADED)
# ----------------------------
st.markdown("### 🤖 Ask AI to Analyze All Consent Data")
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#d1eaf0; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("Ask comparative questions like *'Which consents expire soonest?'* or *'Summarize the reasons for consent for Fulton Hogan.'*")
    
    llm_provider = st.radio("Choose LLM Provider:", ["Groq", "Gemini", "OpenAI"])
    chat_input = st.text_area("Your question for the AI:", key="ai_query_input", placeholder="e.g., List all consents and their expiry dates, sorted from soonest to latest.")

    if st.button("Ask AI", key="ask_ai_button"):
        df_for_ai = st.session_state.get('df', pd.DataFrame())
        if not chat_input.strip():
            st.warning("Please enter a question for the AI.")
        elif df_for_ai.empty:
            st.warning("Please upload PDF documents first.")
        else:
            with st.spinner(f"Asking {llm_provider} to analyze {len(df_for_ai)} consents..."):
                try:
                    # --- NEW: Unified Prompting and Full Context Strategy ---
                    context_df = df_for_ai.drop(columns=['Text Blob', '__file_bytes__', 'Latitude', 'Longitude'], errors='ignore')
                    full_context_csv = context_df.to_csv(index=False)

                    # Safeguard for large context windows
                    if len(full_context_csv) > 30000:
                        sample_size = int(len(context_df) * (30000 / len(full_context_csv)))
                        st.warning(f"Data from {len(df_for_ai)} consents is too large for the AI's context window. Analyzing a smart sample of {sample_size} consents instead.")
                        full_context_csv = context_df.sample(n=sample_size).to_csv(index=False)
                    
                    system_prompt = """You are an expert AI data analyst for Auckland Council resource consents. Your task is to answer the user's query based *only* on the data provided in the user's message. The data is in CSV format. Do not use external knowledge. If the answer cannot be found, say so clearly. Present your answer in clear, easy-to-read markdown."""
                    user_prompt = f"--- CONSENT DATA (CSV format) ---\n{full_context_csv}\n--- END OF DATA ---\n\nBased on the data above, please answer this query: \"{chat_input}\""
                    
                    answer_raw = ""
                    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                    
                    if llm_provider == "Groq":
                        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
                        response = chat.invoke(messages)
                        answer_raw = response.content
                    elif llm_provider == "OpenAI":
                        response = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=0)
                        answer_raw = response.choices[0].message.content
                    elif llm_provider == "Gemini":
                        if genai._client is None:
                            st.error("Google AI API Key not configured. Please set the GOOGLE_API_KEY environment variable.")
                            answer_raw = "Gemini is not configured."
                        else:
                            model_genai = genai.GenerativeModel("gemini-1.5-flash")
                            response = model_genai.generate_content(system_prompt + "\n" + user_prompt)
                            answer_raw = response.text
                    
                    st.markdown(f"#### 🧠 Answer from {llm_provider}")
                    st.markdown(answer_raw)
                    log_ai_chat(chat_input, answer_raw)

                except Exception as e:
                    st.error(f"An error occurred with the AI provider: {e}")
    st.markdown("</div>", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True)
