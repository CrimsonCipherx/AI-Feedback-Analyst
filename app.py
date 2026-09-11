import json
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI Customer-Service & Feedback Analyst",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
.metric-card {padding: 1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.25);}
.small-muted {opacity: .7; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

CATEGORIES = [
    "Product Quality",
    "Shipping/Delivery",
    "Pricing",
    "Customer Service",
    "Staff Behavior",
    "Returns/Refunds",
    "Other",
]
SENTIMENTS = ["Positive", "Neutral", "Negative"]

SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "row_id": {"type": "integer"},
                    "sentiment": {"type": "string", "enum": SENTIMENTS},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "root_cause": {"type": "string"},
                    "suggested_action": {"type": "string"},
                },
                "required": [
                    "row_id",
                    "sentiment",
                    "category",
                    "root_cause",
                    "suggested_action",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a customer-experience analyst for a retail business.
Analyze each customer message independently.

Rules:
- Sentiment must be Positive, Neutral, or Negative.
- Category must be one of: Product Quality, Shipping/Delivery, Pricing,
  Customer Service, Staff Behavior, Returns/Refunds, Other.
- Root cause must explain the likely business reason in one concise sentence.
- Suggested action must be practical and useful to a manager.
- Do not invent facts that are not supported by the customer message.
- Return one result for every row_id provided.
"""

def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Please upload a CSV or XLSX file.")

def find_text_column(df):
    preferred = ["customer_message", "message", "review", "comment", "feedback", "text"]
    lower = {str(c).lower(): c for c in df.columns}
    for p in preferred:
        if p in lower:
            return lower[p]
    # Fall back to the first mostly-text column.
    candidates = [c for c in df.columns if df[c].dtype == "object"]
    if candidates:
        return candidates[0]
    return None

def analyze_batch(client, batch):
    payload = [
        {"row_id": int(r["row_id"]), "customer_message": str(r["customer_message"])}
        for r in batch
    ]
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=SYSTEM_PROMPT,
        input=json.dumps(payload, ensure_ascii=False),
        text={
            "format": {
                "type": "json_schema",
                "name": "customer_feedback_analysis",
                "schema": SCHEMA,
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)["results"]

def generate_summary(client, df):
    stats = {
        "tickets": int(len(df)),
        "sentiment": df["sentiment"].value_counts().to_dict(),
        "categories": df["category"].value_counts().to_dict(),
        "top_actions": df.loc[df["sentiment"] == "Negative", "suggested_action"]
            .value_counts().head(8).to_dict(),
    }
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        instructions=(
            "You are writing a concise executive summary for a retail manager. "
            "Use only the supplied aggregate statistics. Give 3-5 bullet points: "
            "what customers are experiencing, the biggest issue, and prioritized actions. "
            "Do not invent numbers."
        ),
        input=json.dumps(stats, ensure_ascii=False),
    )
    return response.output_text

st.title("📊 AI Customer-Service & Feedback Analyst")
st.caption("Turn raw customer feedback into structured business insights.")

with st.sidebar:
    st.header("Controls")
    api_key = st.text_input(
        "OpenAI API key",
        type="password",
        help="Stored only in this Streamlit session; never commit it to GitHub.",
    )
    model = st.text_input("Model", value=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    batch_size = st.slider("Tickets per AI batch", 5, 50, 20)

    st.divider()
    st.markdown("**Expected input**")
    st.code("customer_message\nThe delivery was two days late.", language="text")
    st.markdown('<div class="small-muted">The app also tries to detect common column names such as review, comment, feedback, or text.</div>', unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload customer feedback",
    type=["csv", "xlsx"],
    help="CSV or Excel file containing a customer-message/review column.",
)

demo_path = Path("data/sample_customer_feedback.csv")

if uploaded is None:
    st.info("No file uploaded — loading the included synthetic retail dataset so the dashboard can be explored immediately.")
    df = pd.read_csv(demo_path)
else:
    try:
        df = load_file(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()

text_col = find_text_column(df)
if text_col is None:
    st.error("I couldn't find a text column. Add a column such as customer_message, review, comment, feedback, or text.")
    st.stop()

df = df.copy()
df = df.dropna(subset=[text_col])
df["customer_message"] = df[text_col].astype(str).str.strip()
df = df[df["customer_message"] != ""].reset_index(drop=True)
df["row_id"] = range(1, len(df) + 1)

st.write(f"**{len(df):,}** customer messages loaded from `{text_col}`.")

if "sentiment" not in df.columns:
    df["sentiment"] = None
if "category" not in df.columns:
    df["category"] = None
if "root_cause" not in df.columns:
    df["root_cause"] = None
if "suggested_action" not in df.columns:
    df["suggested_action"] = None

needs_analysis = df["sentiment"].isna().any()

if st.button("🤖 Analyze with AI", type="primary", disabled=(not api_key)):
    client = OpenAI(api_key=api_key)
    all_results = []
    progress = st.progress(0)
    total_batches = max(1, (len(df) + batch_size - 1) // batch_size)

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size][["row_id", "customer_message"]].to_dict("records")
        try:
            all_results.extend(analyze_batch(client, batch))
        except Exception as e:
            st.error(f"AI analysis failed on batch {i // batch_size + 1}: {e}")
            st.stop()
        progress.progress(min(1.0, (i // batch_size + 1) / total_batches))

    results_df = pd.DataFrame(all_results)
    df = df.drop(columns=["sentiment", "category", "root_cause", "suggested_action"])
    df = df.merge(results_df, on="row_id", how="left")
    st.session_state["analysis"] = df
    st.success(f"Analyzed {len(results_df):,} tickets.")
elif not api_key and needs_analysis:
    st.warning("Add your OpenAI API key in the sidebar to run live AI analysis. The synthetic dataset is included so the UI can still be demonstrated.")

if "analysis" in st.session_state:
    df = st.session_state["analysis"]

if df["sentiment"].notna().any():
    analyzed = df.dropna(subset=["sentiment"]).copy()
    total = len(analyzed)
    negative_pct = (analyzed["sentiment"].eq("Negative").mean() * 100)
    top_category = analyzed["category"].mode().iat[0] if not analyzed["category"].dropna().empty else "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickets analyzed", f"{total:,}")
    c2.metric("Negative sentiment", f"{negative_pct:.1f}%")
    c3.metric("#1 complaint category", top_category)

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment")
        sentiment_counts = analyzed["sentiment"].value_counts().rename_axis("Sentiment").to_frame("Tickets")
        st.bar_chart(sentiment_counts)

    with right:
        st.subheader("Complaint categories")
        cat_counts = analyzed["category"].value_counts().rename_axis("Category").to_frame("Tickets")
        st.bar_chart(cat_counts)

    st.subheader("Customer feedback explorer")
    sentiment_filter = st.multiselect(
        "Filter sentiment",
        SENTIMENTS,
        default=SENTIMENTS,
    )
    category_filter = st.multiselect(
        "Filter category",
        CATEGORIES,
        default=CATEGORIES,
    )
    view = analyzed[
        analyzed["sentiment"].isin(sentiment_filter)
        & analyzed["category"].isin(category_filter)
    ][
        ["row_id", "customer_message", "sentiment", "category", "root_cause", "suggested_action"]
    ]
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download analyzed CSV",
        data=analyzed.to_csv(index=False).encode("utf-8"),
        file_name="analyzed_customer_feedback.csv",
        mime="text/csv",
    )

    st.divider()
    st.subheader("Executive Summary")
    if api_key and st.button("Generate AI executive summary"):
        client = OpenAI(api_key=api_key)
        with st.spinner("Writing management summary..."):
            st.session_state["summary"] = generate_summary(client, analyzed)

    if "summary" in st.session_state:
        st.markdown(st.session_state["summary"])
    else:
        st.caption("Generate an AI executive summary after analysis.")
else:
    st.subheader("Preview")
    st.dataframe(df[["row_id", "customer_message"]], use_container_width=True, hide_index=True)

st.divider()
st.caption("Portfolio project • Synthetic data only • Do not upload confidential customer information.")
