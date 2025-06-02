# Auckland Air Discharge Consent Dashboard - Cleaned & Optimized with Gemini & Groq Fallback

import streamlit as st
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

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
from langchain_groq import ChatGroq

# Load Environment Variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.warning("⚠️ GROQ_API_KEY not set. Groq will not be available.")

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

# ===============================
# Chatbot for Consent Interaction
# ===============================
st.markdown("### 🤖 Ask AI About Consents")
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)

    llm_provider = st.radio("Choose LLM Provider", ["Gemini", "Groq"], horizontal=True)
    chat_input = st.text_area("Search any query:", key="chat_input")

    if st.button("Ask AI"):
        if not chat_input.strip():
            st.warning("Please enter any query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    # Dummy sample context — replace with actual `df[...]` in full app
                    context_sample = [
                        {"Company Name": "ABC Ltd", "Consent Status": "Active", "AUP(OP) Triggers": "E14.1.1", 
                         "Mitigation (Consent Conditions)": "Dust Management Plan", "Expiry Date": "2025-12-31"},
                        {"Company Name": "XYZ Corp", "Consent Status": "Expired", "AUP(OP) Triggers": "E14.2.3", 
                         "Mitigation (Consent Conditions)": "Odour Management Plan", "Expiry Date": "2023-07-01"},
                    ]

                    user_query = f"""
Use the following air discharge consent data to answer the user query.

---
Sample Data:
{context_sample}

---
Query: {chat_input}

Please provide your answer in bullet points.
"""

                    if llm_provider == "Gemini":
                        response = genai.generate_text(model="gemini-pro", prompt=user_query)
                        answer_raw = response.result
                    elif llm_provider == "Groq":
                        if not groq_api_key:
                            raise ValueError("Missing GROQ_API_KEY")
                        llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
                        result = llm.invoke(user_query)
                        answer_raw = result.content if hasattr(result, 'content') else result

                    answer = f"### 🧠 Answer from {llm_provider} AI\n\n{answer_raw}"
                except Exception as e:
                    answer = f"**AI error:** {e} 🚫"
                st.markdown(answer, unsafe_allow_html=False)
    st.markdown("</div>", unsafe_allow_html=True)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True
)
