Auckland Air Discharge Consent Dashboard
Interactive AI-powered dashboard for exploring industrial air discharge consents across Auckland, New Zealand.
This tool enables semantic search, geospatial mapping, and automated metadata extraction from unstructured PDF consent documents.

📊 Features

✅ Upload and process multiple PDF consent documents
📍 Extract and geocode site addresses using Nominatim
🧠 Perform semantic search across consents using BERT-based LLMs (MiniLM, BGE, E5)
📅 Detect issue and expiry dates, and categorize expiring consents
🗺️ Visualize consents on an interactive map with Plotly
📈 View charts of consent status (Active, Expired, Expiring Soon)
🧾 Download CSV summaries and original PDF documents
🧠 Powered by SentenceTransformers (all-MiniLM-L6-v2 and others)
⚡ Built with Streamlit for fast, responsive interaction

===================================

🔍 How It Works
  PDF Parsing: Uses PyMuPDF to extract text.

  Metadata Extraction: Regex rules pull out:
  Company Name
  Consent Numbers
  Site Address
  Issue/Expiry Dates
  Triggered AUP(E14) Rules
  Mitigation Strategies
  Geocoding: Uses geopy (Nominatim) to locate addresses on a map
  Embedding & Semantic Search: Converts full-text blobs into embeddings, allowing query similarity via cosine distance
  Visualizations: Plotly renders:
  Consent status bar charts
  Geolocation markers on a Mapbox map
  
===================================

🧠 Supported LLM Models
  You can switch models from the sidebar:
  all-MiniLM-L6-v2
  multi-qa-MiniLM-L6-cos-v1
  BAAI/bge-base-en-v1.5
  intfloat/e5-base-v2
  
===================================

📥 Input Format
  Upload 1 or more PDF files
  The dashboard automatically parses and extracts relevant consent metadata
  
===================================

🧾 Output Options
  Download metadata as CSV
  Download individual PDF files from within search/map results
  
===================================

⚠️ Limitations
  Address geocoding may occasionally fail if the location format is incomplete
  OCR fallback is not enabled by default (only pure text extraction from PDFs)
  Consent expiry logic is based on simple datetime comparison

  ===================================

👥 Contributors
  Earl Tavera – Data Analytics Developer & Dashboard Architect
  Alana Jacobson-Pepere – LLM Research and Consent Text Parsing
