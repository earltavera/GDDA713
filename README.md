Auckland Air Discharge Consent Dashboard
Welcome to the Auckland Air Discharge Consent Dashboard! This tool allows you to extract, visualize, and analyze data from Air Discharge Resource Consent Decision Reports in PDF format. You can upload multiple reports, view key metadata, monitor consent statuses, and even leverage AI to query the extracted information.

How to Operate the Dashboard
  1. Accessing the Dashboard
  The dashboard is a web application built with Streamlit. Once the application is running, you can access it via your web browser at the provided local   URL (e.g., http://localhost:8501).

2. Dashboard Overview
  At the top of the dashboard, you'll see a banner displaying the current date and time in Auckland, along with the current weather conditions. This       provides immediate local context.

The main section is titled "Welcome to the Auckland Air Discharge Consent Dashboard," which gives a brief introduction to its purpose.

3. Control Panel (Sidebar)
  The primary interactions with the dashboard happen through the Control Panel located on the left sidebar.

Choose Embedding Model:

  This dropdown allows you to select the AI model used for the LLM Semantic Search. Different models might offer varying levels of accuracy and speed      for understanding the meaning behind your search queries.

Options include: all-MiniLM-L6-v2, multi-qa-MiniLM-L6-cos-v1, BAAI/bge-base-en-v1.5, intfloat/e5-base-v2. Experiment to find the one that best suits your needs.

Upload PDF files:

  Click the "Browse files" button to select one or more PDF files from your computer. These should be your Air Discharge Resource Consent Decision         Reports.

The dashboard will automatically process these files upon upload. A progress bar will indicate the status of the file processing, including text extraction and geocoding.

LLM Semantic Search Query:

Enter natural language questions or keywords here to perform a semantic search across all your uploaded documents.

For example, you could ask: "What are the conditions for the consent at 100 Main Street?", or "Show me consents related to dust emissions."

4. Dashboard Sections
  Once you upload PDF files, the main dashboard area will populate with various sections:

Consent Summary Metrics:

This section provides quick numerical summaries:

Total Consents: The total number of unique consent documents processed.

Expiring in 90 Days: The count of consents that will expire within the next 90 days.

Expired: The count of consents that have already expired.

Active: The count of consents that are currently active and not expiring soon.

Consent Status Overview (Chart):

A bar chart visually representing the distribution of consents by their status (Active, Expiring in 90 Days, Expired, Unknown).

Consent Table:

An expandable section that displays a detailed table of all extracted consent data.

Filter by Status: Use the dropdown above the table to filter consents by their Consent Status.

Columns: The table includes key information such as:

File Name

Resource Consent Numbers

Company Name

Address

Issue Date

Expiry Date

Consent Status (Enhanced to include "Expiring in 90 Days")

AUP(OP) Triggers

Reason for Consent

Consent Condition Numbers

Download CSV: A button below the table allows you to download the currently displayed (filtered or unfiltered) consent data as a CSV file.

Consent Map:

An expandable section showing a map of Auckland with markers representing the geocoded addresses of the consents.

Hover over a marker to see basic information about the consent.

Note: If an address cannot be geocoded, it will not appear on the map.

LLM Semantic Search Results:

This section displays the results of your query from the sidebar's "LLM Semantic Search Query" input.

It shows relevant documents based on their semantic similarity to your query, along with a similarity score.

LLM Semantic Search Relevance Threshold: Use the slider to adjust how strict the relevance must be for documents to appear in the search results. A higher threshold means only very similar documents will be shown.

For each result, you'll see the company name, address, triggers, and expiry date.

Download PDF: A "Download PDF" button is provided for each search result, allowing you to download the original PDF file.

5. Ask AI About Consents (Chatbot)
  This interactive AI chatbot allows you to ask more complex questions about the aggregated data or general air discharge consent information.

Choose LLM Provider: Select between Gemini AI and Groq AI as your underlying AI model.

Note: For these AI models to work, you need to have valid API keys configured in your environment variables (GOOGLE_API_KEY for Gemini and GROQ_API_KEY for Groq) or in Streamlit secrets. If keys are missing, the AI will operate in "offline mode" or display an error.

Search any query: Type your question here.

Example Queries:

"Which companies have expired consents?"

"List all consents with E14.1.1 triggers."

"How many active consents are there in Manukau?"

"What is the average expiry duration for consents?"

Ask AI Button: Click this button to send your query to the selected AI model.

The AI's answer will appear below the input box.

Download Chat History (CSV): A button to download a CSV file of all your AI chat interactions.
