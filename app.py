# --------------------------------------------
# Auckland Air Discharge Consent Dashboard
# --------------------------------------------

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
import pytz
import openai  # ✅ Use OpenAI SDK pointed at DeepSeek API

# ------------------------
# API Key Setup
# ------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")  # ✅ DeepSeek API endpoint

# ------------------------
# Streamlit Page Config & Style
# ------------------------
st.set_page_config(page_title="Auckland Air Discharge Consent Dashboard", layout="wide", page_icon="🇳🇿")

# [Rest of your code remains unchanged until the chatbot section]

# Chatbot
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)
    chat_input = st.text_area("Search any query:", key="chat_input")
    if st.button("Ask AI"):
        if not chat_input.strip():
            st.warning("Please enter any query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    context_sample = df[[
                        "Company Name", "Consent Status", "AUP(OP) Triggers", 
                        "Mitigation (Consent Conditions)", "Expiry Date"
                    ]].dropna().head(10).to_dict(orient="records")

                    messages = [
                        {"role": "system", "content": "You are a helpful assistant specialized in environmental compliance and industrial air discharge consents. Use bullet points where possible and highlight key terms in bold."},
                        {"role": "user", "content": f"Data sample: {context_sample}\n\nQuestion: {chat_input}"}
                    ]

                    response = openai.ChatCompletion.create(
                        model="deepseek-chat",
                        messages=messages,
                        max_tokens=500,
                        temperature=0.7
                    )
                    answer_raw = response["choices"][0]["message"]["content"]
                    answer = f"""### 🧠 Answer from AI\n\n{answer_raw}"""
                except Exception as e:
                    answer = f"**AI error:** {e}"
                st.markdown(answer, unsafe_allow_html=False)
                log_ai_chat(chat_input, answer_raw)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True
)
