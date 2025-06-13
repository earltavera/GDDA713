# Auckland Air Discharge Consent Dashboard - Complete with Gemini, OpenAI, and Groq Chatbot

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
    if expiry_date is None:
        return "Unknown"
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
        r"expires (\d{1,} years [A-Za-z]+[.?1])"  # CORRECTED THIS LINE
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
        r"(?<=Conditions).*?(?=Advice notes)",
        r"(?<=Specific conditions - Air Discharge DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions -)",
        r"(?<=Air Quality conditions).*?(?=Wastewater Discharge conditions)",
        r"(?<=Air Discharge Permit Conditions).*?(?=E. Definitions)",
        r"(?<=Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=DIS\d{5,}(?:-\w+)?\b)",
        r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?<=Conditions Specific to air quality).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=regional discharge DIS\d{5,}(?:-w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?<=Specific conditions - Air Discharge DIS\d{5,}\b).*?(?=Advice notes)",
        r"(?<=Air discharge - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=Advice notes)",
        r"(?<=DIS\d{5,}(?:-\w+)?\b - Specific conditions).*?(?=DIS\d{5,}(?:-\w+)?\b - Specific conditions)",
        r"(?<=Specific Conditions - DIS\d{5,}(?:-\w+)?\b (s15 Air Discharge permit)).*?(?=Advice notes)",
        r"(?<=Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b Only).*?(?=Advice notes)",
        r"(?<=Conditions relevant to Air Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Specific Conditions -)",
        r"(?<=SPECIFIC CONDITIONS - DISCHARGE TO AIR DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Conditions relevant to Discharge Permit DIS\d{5,}(?:-\w+)?\b only).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit (DIS\d{5,}(?:-\w+)?\b)).*?(?=Advice notes)",
        r"(?<=Specific conditions - DIS\d{5,}(?:-\w+)?\b (air)).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specifc conditions)",
        r"(?<=Attachment 1: Consolidated conditions of consent as amended).*?(?=Advice notes)",
        r"(?<=Specific Air Discharge Conditions).*?(?=Advice notes)",
        r"(?<=Specific conditions - Discharge to Air: DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - discharge permit (air discharge) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Air Discharge Limits).*?(?= Acoustic Conditions)",
        r"(?<=Specific conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit (s15) DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific conditions - air discharge permit DIS\d{5,}(?:-\w+)?\b).*?(?=Secific conditions)",
        r"(?<=Specific conditions relating to Air discharge permit - DIS\d{5,}(?:-\w+)?\b).*?(?=General Advice notes)",
        r"(?<=Specific conditions - Discharge permit (s15) - DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=Specific Conditions - discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?<=Specific conditions - Discharge to air: DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?<=Attachement 1: Consolidated conditions of consent as amended).*?(?=Resource Consent Notice of Works Starting)",
        r"(?<=Specific conditions - Air Discharge consent - DIS\d{5,}(?:-\w+)?\b).*?(?=Specific conditions)",
        r"(?<=Specific conditions - Discharge consent DIS\d{5,}(?:-\w+)?\b).*?(?=Advice notes)",
        r"(?<=DIS\d{5,}(?:-\w+)?\b - Air Discharge).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?<=DIS\d{5,}(?:-\w+)?\b & DIS\d{5,}(?:-\w+)?\b).*?(?=).*?(?=SUB\d{5,}\b) - Subdivision",
        r"(?<=Specific conditions - Discharge Permit DIS\d{5,}(?:-\w+)?\b).*?(?=Advice Notes - General)",
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


def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer_raw):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # CORRECTED VARIABLE NAME FROM 'answer' to 'answer_raw'
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer_raw}
    file_path = "ai_chat_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["Timestamp", "Question", "Answer"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

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
    return None

# --------------------
# Sidebar & Model Loader
# --------------------

st.sidebar.markdown("""
    <h2 style='color:#2c6e91; font-family:Segoe UI, Roboto, sans-serif;'>
        Control Panel
    </h2>
""", unsafe_allow_html=True)

model_name = st.sidebar.selectbox("Choose LLM model:", [
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2"
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
if uploaded_files:
    all_data = []
    # Note: OCR fallback requires pytesseract and pdf2image.
    # To keep this script runnable without extra installs, the OCR part is omitted.
    # You can re-add it if those libraries are installed in your environment.
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

        # Normalize dates after loading into DataFrame
        df["Issue Date"] = pd.to_datetime(df["Issue Date"], errors='coerce', dayfirst=True)
        df["Expiry Date"] = pd.to_datetime(df["Expiry Date"], errors='coerce', dayfirst=True)

        df["Consent Status Enhanced"] = df["Consent Status"]
        df.loc[
            (df["Consent Status"] == "Active") &
            (df["Expiry Date"] > datetime.now()) &
            (df["Expiry Date"] <= datetime.now() + timedelta(days=90)),
            "Consent Status Enhanced"
        ] = "Expiring in 90 Days"

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
            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_data, "filtered_consents.csv", "text/csv")

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
                    height=500,
                    color="Consent Status Enhanced",
                    color_discrete_map=color_map
                )
                fig.update_traces(marker=dict(size=12))
                fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig, use_container_width=True)

        # Enhanced Semantic Search Section
        with st.expander("Semantic Search Results", expanded=True):
            if query_input:
                st.info("Running enhanced semantic + structured search...")

                def normalize(text):
                    return re.sub(r"\s+", " ", str(text).lower())

                corpus = (
                    df["Company Name"].fillna("") + " | " +
                    df["Address"].fillna("") + " | " +
                    df["AUP(OP) Triggers"].fillna("") + " | " +
                    df["Mitigation (Consent Conditions)"].fillna("") + " | " +
                    df["Reason for Consent"].fillna("") + " | " +
                    df["Consent Conditions"].fillna("") + " | " +
                    df["Resource Consent Numbers"].fillna("") + " | " +
                    df["Text Blob"].fillna("")
                ).apply(normalize).tolist()

                query_input_norm = normalize(query_input)
                corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
                query_embedding = model.encode(query_input_norm, convert_to_tensor=True)

                if "e5" in model_name or "bge" in model_name:
                    scores = query_embedding @ corpus_embeddings.T
                else:
                    scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

                top_k = scores.argsort(descending=True)[:5]

                keyword_matches = df[
                    df["Address"].str.contains(query_input, case=False, na=False) |
                    df["Resource Consent Numbers"].str.contains(query_input, case=False, na=False) |
                    df["Reason for Consent"].str.contains(query_input, case=False, na=False)
                ]

                if top_k is not None and any(scores[top_k] > 0.3):
                    st.success("Top semantic results:")
                    for i, idx in enumerate(top_k):
                        row = df.iloc[idx.item()]
                        st.markdown(f"**{i+1}. {row['Company Name']}**")
                        st.markdown(f"- 📍 **Address**: {row['Address']}")
                        st.markdown(f"- 🔢 **Consent Number**: {row['Resource Consent Numbers']}")
                        st.markdown(f"- 📜 **Reason**: {row['Reason for Consent']}")
                        st.markdown(f"- ⏳ **Expiry**: {row['Expiry Date'].strftime('%d-%m-%Y') if pd.notnull(row['Expiry Date']) else 'Unknown'}")
                        safe_filename = clean_surrogates(row['__file_name__'])
                        st.download_button(
                            label=f"📄 Download PDF: {safe_filename}",
                            data=row['__file_bytes__'],
                            file_name=safe_filename,
                            mime="application/pdf",
                            key=f"download_semantic_{i}"
                        )
                        st.markdown("---")

                elif not keyword_matches.empty:
                    st.info("Showing keyword-based results:")
                    for i, row in keyword_matches.head(5).iterrows():
                        st.markdown(f"**{row['Company Name']}**")
                        st.markdown(f"- 📍 **Address**: {row['Address']}")
                        st.markdown(f"- 🔢 **Consent Number**: {row['Resource Consent Numbers']}")
                        st.markdown(f"- ⏳ **Expiry**: {row['Expiry Date'].strftime('%d-%m-%Y') if pd.notnull(row['Expiry Date']) else 'Unknown'}")
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
                    st.warning("No strong semantic or keyword matches found.")

# ----------------------------
# Ask AI About Consents Chatbot
# ----------------------------
st.markdown("### 🤖 Ask AI About Consents")
with st.expander("Ask AI About Consents", expanded=True):
    st.markdown("""<div style="background-color:#ff8da1; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
    st.markdown("**Ask anything about air discharge consents** (e.g. triggers, expiry, mitigation, or general trends)", unsafe_allow_html=True)

    llm_provider = st.radio("Choose LLM Provider", ["Gemini", "OpenAI", "Groq", "HuggingFace"], horizontal=True)
    chat_input = st.text_area("Search any query:", key="chat_input")

    if st.button("Ask AI"):
        if not chat_input.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    if "df" in locals() and not df.empty:
                        context_sample = df[[
                            "Company Name", "Consent Status", "AUP(OP) Triggers",
                            "Mitigation (Consent Conditions)", "Expiry Date"
                        ]].dropna().head(5).to_dict(orient="records")
                    else:
                        st.warning("No documents uploaded. AI is answering with general knowledge.")
                        context_sample = [{"Company Name": "ABC Ltd", "Consent Status": "Active", "AUP(OP) Triggers": "E14.1.1", "Mitigation (Consent Conditions)": "Dust Management Plan", "Expiry Date": "2025-12-31"}]

                    user_query = f"""
Use the following air discharge consent data to answer the user query.

---
Sample Data:
{context_sample}

---
Query: {chat_input}

Please provide your answer in bullet points.
"""
                    answer_raw = ""
                    if llm_provider == "Gemini":
                        model_genai = genai.GenerativeModel("gemini-pro")
                        response = model_genai.generate_content(user_query)
                        answer_raw = response.text
                    elif llm_provider == "OpenAI":
                        messages = [
                            {"role": "system", "content": "You are a helpful assistant for environmental consents."},
                            {"role": "user", "content": user_query}
                        ]
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=messages,
                            max_tokens=500,
                            temperature=0.7
                        )
                        answer_raw = response.choices[0].message.content
                    elif llm_provider == "Groq":
                        chat = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
                        system_message = "You are an environmental compliance assistant. Answer based only on the provided data."
                        groq_response = chat.invoke([
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_query}
                        ])
                        answer_raw = groq_response.content if hasattr(groq_response, 'content') else str(groq_response)
                    elif llm_provider == "HuggingFace":
                        # The hf_chatbot function was not defined, so this section is disabled.
                        # You would need to implement the model loading and pipeline for this to work.
                        # For example: from transformers import pipeline
                        # hf_chatbot = pipeline("text-generation", model="some-model")
                        st.warning("HuggingFace provider is not implemented in this version.")
                        answer_raw = "This AI provider is currently unavailable."

                    st.markdown(f"### 🧠 Answer from {llm_provider} AI\n\n{answer_raw}")
                    # ADDED LOGGING CALL
                    if answer_raw and "unavailable" not in answer_raw:
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
