# AI Customer-Service & Feedback Analyst

A portfolio project of mine that connects real-world retail experience with Python, AI, data analysis, and business intelligence.

 What it does:

1. Uploads a CSV/XLSX of customer feedback.
2. Detects a likely feedback/message column.
3. Sends feedback to the OpenAI API in batches.
4. Uses Structured Outputs so the AI returns predictable fields:
   - Sentiment
   - Category
   - Root cause
   - Suggested action
5. Displays KPIs and charts in Streamlit.
6. Lets managers filter feedback and download the enriched CSV.
7. Generates an AI executive summary.

How to Run locally:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

Add your OpenAI API key in the sidebar when the app starts.

Optional environment variable:

```bash
OPENAI_MODEL=gpt-5-mini
```

## Input format

The app accepts CSV/XLSX. Ideally include:

```text
customer_message
"The delivery arrived two days late."
"The headphones stopped working after one week."
```

It also recognizes common alternatives such as `review`, `comment`, `feedback`, `message`, and `text`.


## Architecture

```text
CSV/XLSX
   |
   v
Streamlit Upload
   |
   v
Pandas cleaning
   |
   v
OpenAI Responses API
Structured JSON Schema
   |
   v
Enriched Pandas DataFrame
   |
   +--> KPIs
   +--> Charts
   +--> Filterable table
   +--> CSV export
   +--> Executive summary
```

## Next upcoming upgrades

- Add confidence scores.
- Add monthly trend analysis when a date column exists.
- Add branch/store analysis when location data exists.
- Add Arabic + English feedback support.
- Add duplicate/near-duplicate complaint detection.
- Add Power BI export.
- Add automated evaluation against a human-labeled test set.
- Add unit tests and an evaluation dataset.
- Add cost/token tracking.
