import streamlit as st
import pdfplumber
import pandas as pd
from transformers import pipeline
import torch

# UI Configuration
st.set_page_config(page_title="Legal Audit Dashboard", layout="wide")

# Custom CSS for Professional/Light Theme
# Custom CSS for Professional Legal Tech Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background */
    .stApp {
        background-color: #f8f9fc;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #94a3b8;
        font-size: 1.1rem;
    }

    /* File Uploader Container */
    .uploadedFile {
        border: 2px dashed #cbd5e1;
        background-color: #ffffff;
        padding: 40px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 30px;
        transition: all 0.3s ease;
    }
    .uploadedFile:hover {
        border-color: #3b82f6;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* Cards Common */
    .clause-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    .clause-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    /* High Risk Card Specifics */
    .risk-card {
        border-left: 6px solid #ef4444;
    }
    .risk-badges {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .risk-badge {
        background: #fef2f2;
        color: #b91c1c;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .risk-score {
        background: #ef4444;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Safe Card Specifics */
    .safe-card {
        border-left: 6px solid #10b981;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .check-icon {
        color: #10b981;
        font-size: 1.2rem;
    }

    /* Verdict Box */
    .verdict-box {
        background: linear-gradient(135deg, #eff6ff 0%, #ffffff 100%);
        border: 1px solid #bfdbfe;
        padding: 2rem;
        border-radius: 16px;
        margin-top: 2rem;
        color: #1e3a8a;
    }
    .verdict-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #1e40af;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipelines():
    # 'nlpaueb/legal-bert-base-uncased' performed poorly in verification (0.27 score on clear risks).
    # Switching to 'facebook/bart-large-mnli' for robust zero-shot classification.
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
    return classifier, summarizer

def extract_text(feed):
    text = ""
    with pdfplumber.open(feed) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    return text

def get_negotiation_tip(label):
    tips = {
        "Adversarial/Risky Trap": "🚩 <b>Negotiation Tip:</b> Request specific definition or removal of ambiguous terms.",
        "Unfair Arbitration": "⚖️ <b>Negotiation Tip:</b> Propose mutual arbitration or a neutral venue (e.g., AAA rules).",
        "Hidden Liability": "🛡️ <b>Negotiation Tip:</b> Explicitly request mutual liability or a cap on damages (e.g., 1x contract value)."
    }
    return tips.get(label, "Review carefully.")

def main():
    # Header
    st.markdown("""
        <div class="main-header">
            <h1>⚖️ Legal Audit Dashboard</h1>
            <p>AI-Powered Contract Risk Analysis & Auditor</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar / Setup
    with st.spinner("Initializing Neural Engines..."):
        classifier, summarizer = load_pipelines()

    # File Uploader
    st.markdown('<div class="uploadedFile"><h3 style="color: #475569; margin:0;">📄 Upload Contract PDF</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

    if uploaded_file:
        text = extract_text(uploaded_file)
        if not text:
            st.error("Could not extract text from PDF.")
            return

        # Pre-process
        clauses = [c.strip() for c in text.replace('\n', '.').split('.') if len(c.strip()) > 20]
        
        # Limit clauses
        if len(clauses) > 50:
            st.info(f"⚡ Analyzing first 50 clauses of {len(clauses)} for rapid prototype...")
            clauses = clauses[:50]

        risks = []
        safe_points = []
        
        candidate_labels = ["Adversarial/Risky Trap", "Unfair Arbitration", "Hidden Liability", "Standard Clause", "Beneficial Clause"]
        risk_labels = ["Adversarial/Risky Trap", "Unfair Arbitration", "Hidden Liability"]

        progress_bar = st.progress(0)
        
        for i, clause in enumerate(clauses):
            progress_bar.progress((i + 1) / len(clauses))
            
            result = classifier(clause, candidate_labels)
            top_label = result['labels'][0]
            score = result['scores'][0]

            if top_label in risk_labels and score > 0.35:
                risks.append({
                    "clause": clause,
                    "label": top_label,
                    "score": score,
                    "tip": get_negotiation_tip(top_label)
                })
            elif score > 0.5: 
                safe_points.append(clause)

        # Remove progress bar
        progress_bar.empty()

        # Risk Density Calculation
        risk_density = len(risks) / len(clauses) if clauses else 0

        # Columns
        col1, col2 = st.columns([1, 1.2]) 
        
        with col1:
            st.markdown(f"### ✅ Verified Safe ({len(safe_points)})")
            if safe_points:
                for p in safe_points[:8]:
                    st.markdown(f"""
                    <div class="clause-card safe-card">
                        <div class="check-icon">✓</div>
                        <div style="color: #374151; font-size: 0.95rem;">{p}</div>
                    </div>
                    """, unsafe_allow_html=True)
                if len(safe_points) > 8:
                    st.caption(f"...and {len(safe_points)-8} more safe clauses.")
            else:
                st.info("No explicitly safe points found.")

        with col2:
            st.markdown(f"### ⚠️ Risk Analysis ({len(risks)} detected)")
            
            # Key Metric
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Risk Density", f"{risk_density:.1%}", delta=None, help="Percentage of risky clauses")
            
            st.markdown("---")

            if risks:
                for r in risks:
                    st.markdown(f"""
                    <div class="clause-card risk-card">
                        <div class="risk-badges">
                            <span class="risk-badge">{r['label']}</span>
                            <span class="risk-score">Confidence: {r['score']:.0%}</span>
                        </div>
                        <div style="font-family: 'Inter', sans-serif; color: #1f2937; margin-bottom: 0.8rem; line-height: 1.5;">
                            "{r['clause']}"
                        </div>
                        <div style="background: #eff6ff; padding: 0.75rem; border-radius: 8px; font-size: 0.9rem; color: #1e40af;">
                            {r['tip']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.success("No high-risk clauses detected. The contract looks standard.")

        # Executive Verdict Section
        st.markdown("___")
        with st.spinner("🤖 Writing Executive Summary..."):
            summary_input = text[:4000] 
            summary_res = summarizer(summary_input, max_length=150, min_length=40, do_sample=False)
            summary_text = summary_res[0]['summary_text']
        
        st.markdown(f"""
        <div class="verdict-box">
            <div class="verdict-title">📋 Executive AI Verdict</div>
            <div style="font-size: 1.1rem; line-height: 1.6;">{summary_text}</div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
