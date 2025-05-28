# Auckland Air Discharge Consent Dashboard - Gemini Only Version with Model Selector

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

# ------------------------
# API Key Setup
# ------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="\U0001F1F3\U0001F1FF")

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
# Utility Functions
# ------------------------
def parse_mixed_date(date_str):
    formats = ["%d-%m-%Y", "%d/%m/%Y", "%d %B %Y", "%d %b %Y"]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except (ValueError, TypeError):
            continue
    return None

def check_expiry(expiry_date):
    if expiry_date is None:
        return "Unknown"
    return "Expired" if expiry_date < datetime.now() else "Active"

def geocode_address(address):
    geolocator = Nominatim(user_agent="air_discharge_dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(address)
    return (location.latitude, location.longitude) if location else (None, None)

def extract_metadata(text):
    rc_matches = re.findall(r"Application number[:\s]*([\w/-]+)", text, re.IGNORECASE) or re.findall(r"RC[0-9]{5,}", text)
    rc_str = "".join(dict.fromkeys(rc_matches))
    company_str = "".join(dict.fromkeys(re.findall(r"Applicant:\s*(.+?)(?=\s*Site address)", text)))
    address_str = "".join(dict.fromkeys(re.findall(r"Site address:\s*(.+?)(?=\s*Legal description)", text)))

    issue_str = "".join(dict.fromkeys(re.findall(r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})", text) +
                                       re.findall(r"Date:\s*(\d{1,2}[/\\-]\d{1,2}[/\\-]\d{2,4})", text)))
    issue_date = parse_mixed_date(issue_str)

    expiry_str = "".join(dict.fromkeys(re.findall(r"shall expire on (\d{1,2} [A-Za-z]+ \d{4})", text) +
                                        re.findall(r"expires on (\d{1,2} [A-Za-z]+ \d{4})", text)))
    expiry_date = parse_mixed_date(expiry_str)

    triggers_str = " ".join(dict.fromkeys(re.findall(r"E\d+\.\d+\.\d+", text) +
                                          re.findall(r"E\d+\.\d+.", text) +
                                          re.findall(r"NES:STO", text) +
                                          re.findall(r"NES:AQ", text)))

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

def ask_ai_with_fallback(question, context_sample, model_name="gemini-pro"):
    try:
        prompt = f"""
You are an expert assistant in environmental regulation. Summarize and extract key insights from the sample consent data and answer the user's question clearly.

Instructions:
- Use **bullet points** for multiple items
- Highlight important terms in **bold**
- If dates, rules (e.g., E14.1.1), or mitigation strategies are relevant, include them
- Be concise and professional

--- Sample Consent Data ---
{context_sample}

--- User Question ---
{question}
"""
        gemini_model = genai.GenerativeModel(model_name)
        response = gemini_model.generate_content(prompt)
        return response.text, f"Gemini ({model_name})"
    except Exception as e:
        return f"Gemini AI failed: {e}", "Error"

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def render_chatbot_section(uploaded_files, key_suffix="", gemini_model_name="gemini-pro"):
    chat_icon = "🤖"
    with st.expander("Ask AI About Consents", expanded=True):
        st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
        st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)

        chat_input = st.text_area("Search any query:", key=f"chat_input_ai_{key_suffix}")

        if st.button("Ask AI", key=f"ask_ai_button_{key_suffix}"):
            if not chat_input.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("AI is thinking..."):
                    try:
                        df_context = []
                        if uploaded_files:
                            for file in uploaded_files:
                                file_bytes = file.read()
                                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                                    text = doc[0].get_text()
                                    metadata = extract_metadata(text)
                                    metadata["__file_name__"] = file.name
                                    metadata["__file_bytes__"] = file_bytes
                                    df_context.append(metadata)

                        if not df_context:
                            st.warning("No data found to build context for AI.")
                            return

                        df_context = pd.DataFrame(df_context)
                        corpus = df_context["Text Blob"].tolist()
                        corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
                        query_embedding = model.encode(chat_input, convert_to_tensor=True)
                        scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
                        top_k = scores.argsort(descending=True)[:3]
                        top_matches = df_context.iloc[[i.item() for i in top_k]]

                        context_sample = top_matches[[
                            "Company Name", "Consent Status", "AUP(OP) Triggers",
                            "Mitigation (Consent Conditions)", "Expiry Date"
                        ]].dropna().to_dict(orient="records")

                        response_text, model_used = ask_ai_with_fallback(chat_input, context_sample, model_name=gemini_model_name)

                        if model_used == "Error":
                            st.error(response_text)
                            return

                        st.markdown(f"""
                            <h3 style='font-size:1.3em'>{chat_icon} Answer from AI <span style='color:#007bff'>(<strong>{model_used}</strong>)</span></h3>
                            <div style='padding-top:0.5em'>{response_text}</div>
                        """, unsafe_allow_html=True)

                        st.markdown("---")
                        st.subheader("📌 Related PDF Downloads")
                        for i, (_, row) in enumerate(top_matches.iterrows(), 1):
                            file_label = clean_surrogates(row['__file_name__'])
                            st.markdown(f"**{i}. {row['Company Name']}** — *{row['Address']}*")
                            st.download_button(
                                label=f"📄 Download PDF: {file_label}",
                                data=row["__file_bytes__"],
                                file_name=file_label,
                                mime="application/pdf",
                                key=f"download_chat_{key_suffix}_{i}"
                            )
                            st.markdown("---")

                    except Exception as err:
                        st.error(f"AI failed: {err}")
        st.markdown("</div>", unsafe_allow_html=True)

# ------------------------
# Sidebar Controls
# ------------------------
st.sidebar.title("Control Panel")
model_name = st.sidebar.selectbox("Embedding Model", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
])

uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
query_input = st.sidebar.text_input("Semantic Search Query")

# Gemini model selector
gemini_model_name = st.sidebar.selectbox("Choose Gemini Model:", ["gemini-pro", "gemini-1.5-pro"], index=1)

@st.cache_resource
def load_model(name):
    return SentenceTransformer(name)

model = load_model(model_name)

# ------------------------
# Top Banner
# ------------------------
nz_time = datetime.now(pytz.timezone("Pacific/Auckland"))
today = nz_time.strftime("%A, %d %B %Y")
current_time = nz_time.strftime("%I:%M %p")
weather = get_auckland_weather()

st.markdown(f"""
    <div style='text-align:center; padding:12px; font-size:1.2em; background-color:#656e6b;
                border-radius:10px; margin-bottom:15px; font-weight:500; color:white;'>
        🗓️ <strong>{today}</strong> &nbsp;&nbsp;&nbsp; ⏰ <strong>{current_time}</strong> &nbsp;&nbsp;&nbsp; 🌦️ <strong>{weather}</strong> &nbsp;&nbsp;&nbsp; 📍 <strong>Auckland</strong>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='color:#2c6e91; text-align:center; font-size:2.7em; font-family: Quicksand, sans-serif;'>
        Auckland Air Discharge Consent Dashboard
    </h1>
""", unsafe_allow_html=True)

# ------------------------
# Chatbot Interface
# ------------------------
render_chatbot_section(uploaded_files, key_suffix="main", gemini_model_name=gemini_model_name)

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True
)
