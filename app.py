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

# File Processing with OCR Fallback
if 'uploaded_files' in locals() and uploaded_files:
    all_data = []
    for file in uploaded_files:
        try:
            file_bytes = file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                text = "\n".join(page.get_text() for page in doc)

            if not text.strip():
                st.warning(f"{file.name} appears to be image-based. Using OCR...")
                images = convert_from_bytes(file_bytes)
                text = "\n".join(pytesseract.image_to_string(img) for img in images)

            data = extract_metadata(text)
            data["__file_name__"] = file.name
            data["__file_bytes__"] = file_bytes
            all_data.append(data)
        except Exception as e:
            st.error(f"Error processing {file.name}: {e}")


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
    rc_matches = re.findall(r"Application number[:\s]*([\w/-]+)", text, re.IGNORECASE) or re.findall(r"RC[0-9]{5,}", text)
    rc_str = "".join(dict.fromkeys(rc_matches))

    company_str = "".join(dict.fromkeys(re.findall(r"Applicant:\s*(.+?)(?=\s*Site address)", text)))
    address_str = "".join(dict.fromkeys(re.findall(r"Site address:\s*(.+?)(?=\s*Legal description)", text)))

    issue_str = "".join(dict.fromkeys(
        re.findall(r"Date:\s*(\d{1,2} [A-Za-z]+ \d{4})", text) +
        re.findall(r"Date:\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text)
    ))
    issue_date = parse_mixed_date(issue_str)

    expiry_str = "".join(dict.fromkeys(
        re.findall(r"shall expire on (\d{1,2} [A-Za-z]+ \d{4})", text) +
        re.findall(r"expires on (\d{1,2} [A-Za-z]+ \d{4})", text)
    ))
    expiry_date = parse_mixed_date(expiry_str)

    triggers_str = " ".join(dict.fromkeys(
        re.findall(r"E\d+\.\d+\.\d+", text) +
        re.findall(r"E\d+\.\d+.", text) +
        re.findall(r"NES:STO", text) +
        re.findall(r"NES:AQ", text)
    ))

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

def clean_surrogates(text):
    return text.encode('utf-16', 'surrogatepass').decode('utf-16', 'ignore')

def log_ai_chat(question, answer_raw):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"Timestamp": timestamp, "Question": question, "Answer": answer}
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

    llm_provider = st.radio("Choose LLM Provider", ["Gemini", "OpenAI", "Groq"], horizontal=True)
    chat_input = st.text_area("Search any query:", key="chat_input")

    if st.button("Ask AI"):
        if not chat_input.strip():
            st.warning("Please enter any query.")
        else:
            with st.spinner("AI is thinking..."):
                try:
                    if "df" in locals():
                        context_sample = df[[
                            "Company Name", "Consent Status", "AUP(OP) Triggers",
                            "Mitigation (Consent Conditions)", "Expiry Date"
                        ]].dropna().head(5).to_dict(orient="records")
                    else:
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
                        model = genai.GenerativeModel("gemini-pro")
                        response = model.generate_content(user_query)
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

                    answer = f"### 🧠 Answer from {llm_provider} AI\n\n{answer_raw}"
                except Exception as e:
                    answer = f"**AI error:** {e}"
                st.markdown(answer, unsafe_allow_html=False)
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: orange; font-size: 0.9em;'>"
    "Built by Earl Tavera & Alana Jacobson-Pepere | Auckland Air Discharge Intelligence © 2025"
    "</p>",
    unsafe_allow_html=True
)
