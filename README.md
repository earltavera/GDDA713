# Auckland Air Discharge Consent Dashboard

A powerful Streamlit dashboard for analyzing industrial air discharge consents in Auckland. Supports both local folder and file upload modes. Features OCR, BERT-powered semantic search, visual summaries, and CSV export.

---

## 🚀 Features

- 📁 Load PDFs from local folder or upload directly
- 🔍 BERT-based semantic search over consent documents
- 🧾 OCR fallback for scanned PDFs using Tesseract
- 📊 Interactive visual summaries (Altair + Streamlit)
- 📈 Filters by industry, pollutant, suburb
- 📥 Download filtered results as CSV or PDFs
- 🧠 Auto-detect consent metadata like:
  - Industry
  - Pollutants
  - Consent / Expiry Dates
  - Mitigation Measures
  - Consultant info

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

> Note: `pytesseract` and `pdf2image` require system-level dependencies:
- **Tesseract OCR** (https://github.com/tesseract-ocr/tesseract)
- **Poppler** (https://github.com/oschwartz10612/poppler-windows or `brew install poppler` on macOS)

---

## 🖥️ Running the App

```bash
streamlit run air_dashboard_ocr_enabled.py
```

---

## 📁 Upload Modes

Choose one:
1. **📂 Folder Path**: Reads all PDFs in a local directory (use on your machine).
2. **📄 Upload Files**: Drag-and-drop PDFs for processing (works in Streamlit Cloud).

---

## 📄 Example Query

Search bar accepts natural language:
```text
Mitigation for dust in South Auckland
```

---

## 👥 Authors

- Earl Tavera
- Alana Jacobson‑Pepere

---

## 📝 License

MIT License — use freely, credit the authors!
