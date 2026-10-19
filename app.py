"""TenderGuard — Intelligent Tender Risk Detection and Contradiction Analyzer.
Streamlit Web Application Interface with User Authentication, Clean Typography, and Modern UI/UX.
"""
import io
import os
import time
from pathlib import Path

# Ensure OpenMP setting is active before imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import streamlit as st

from tenderguard.config import (
    DATA_DIR,
    DEFAULT_CONFIG,
    DEFAULT_DB_PATH,
    TENDER_CATEGORIES,
)
from tenderguard.db.session import DatabaseManager
from tenderguard.evaluation.dataset import load_benchmark_dataset
from tenderguard.evaluation.evaluator import BenchmarkEvaluator
from tenderguard.export.report_generator import ReportGenerator
from tenderguard.pipeline import TenderPipeline
from tenderguard.review.review_manager import REVIEW_STATUS_CHOICES, ReviewManager

# Page configuration
st.set_page_config(
    page_title="TenderGuard — Tender Risk & Contradiction Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System CSS
st.markdown(
    """
    <style>
    /* Clean System Typography & Container Sizing */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: #0F172A;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1280px;
    }

    /* Page Header */
    .tg-header {
        margin-bottom: 1.5rem;
    }
    .tg-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .tg-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        line-height: 1.5;
    }

    /* Cards & Containers */
    .tg-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.04), 0 1px 2px -1px rgba(0, 0, 0, 0.04);
        margin-bottom: 1rem;
    }

    .tg-hero-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .tg-feature-row {
        display: flex;
        align-items: flex-start;
        padding: 0.85rem 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .tg-feature-row:last-child {
        border-bottom: none;
    }
    .tg-feature-icon {
        font-size: 1.25rem;
        margin-right: 0.85rem;
        flex-shrink: 0;
    }
    .tg-feature-title {
        font-weight: 600;
        font-size: 0.92rem;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .tg-feature-desc {
        font-size: 0.84rem;
        color: #64748B;
        line-height: 1.45;
    }

    /* Badges & Status Pills */
    .tg-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.22rem 0.65rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        line-height: 1;
    }
    .badge-high {
        background-color: #FFF1F2;
        color: #BE123C;
        border: 1px solid #FFE4E6;
    }
    .badge-medium {
        background-color: #FFFBEB;
        color: #B45309;
        border: 1px solid #FEF3C7;
    }
    .badge-low {
        background-color: #F0F9FF;
        color: #0369A1;
        border: 1px solid #E0F2FE;
    }
    .badge-confirmed {
        background-color: #ECFDF5;
        color: #047857;
        border: 1px solid #D1FAE5;
    }
    .badge-pending {
        background-color: #F8FAFC;
        color: #475569;
        border: 1px solid #E2E8F0;
    }
    .badge-false-pos {
        background-color: #F1F5F9;
        color: #64748B;
        border: 1px solid #CBD5E1;
    }
    .badge-clarification {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .badge-resolved {
        background-color: #EFF6FF;
        color: #1D4ED8;
        border: 1px solid #DBEAFE;
    }

    .privacy-pill {
        display: inline-flex;
        align-items: center;
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 1.25rem;
    }

    /* Clause & Evidence Boxes */
    .tg-clause-box-a {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #6366F1;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #1E293B;
    }
    .tg-clause-box-b {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #1E293B;
    }
    .tg-clause-box-neutral {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        font-size: 0.88rem;
        line-height: 1.6;
        color: #1E293B;
    }

    /* Sidebar User Profile */
    .tg-user-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem;
        margin-bottom: 0.75rem;
    }
    .tg-user-name {
        font-weight: 700;
        font-size: 0.95rem;
        color: #0F172A;
    }
    .tg-user-role {
        font-size: 0.78rem;
        font-weight: 500;
        color: #4F46E5;
    }
    .tg-user-org {
        font-size: 0.76rem;
        color: #64748B;
        margin-top: 2px;
    }

    /* Streamlit Metric Customization */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.76rem !important;
        font-weight: 600 !important;
        color: #64748B !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }

    /* Streamlit Expander Customization */
    div[data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
    }
    div[data-testid="stExpander"] summary {
        font-weight: 600;
        font-size: 0.92rem;
        color: #1E293B;
        padding: 0.75rem 1rem !important;
    }

    /* Streamlit Tabs Customization */
    button[data-baseweb="tab"] {
        font-size: 0.9rem !important;
        font-weight: 500 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_priority_badge(priority: str) -> str:
    """Return styled HTML pill badge for priority."""
    p = (priority or "low").lower()
    return f"<span class='tg-badge badge-{p}'>{priority} Priority</span>"


def render_status_badge(status: str) -> str:
    """Return styled HTML pill badge for review status."""
    s_map = {
        "Confirmed Risk": "badge-confirmed",
        "Pending Review": "badge-pending",
        "False Positive": "badge-false-pos",
        "Clarification Needed": "badge-clarification",
        "Resolved / Dismissed": "badge-resolved",
    }
    css_class = s_map.get(status, "badge-pending")
    return f"<span class='tg-badge {css_class}'>{status}</span>"


def get_database_manager() -> DatabaseManager:
    """Return configured DatabaseManager instance."""
    return DatabaseManager(DEFAULT_DB_PATH)


@st.cache_resource
def get_pipeline() -> TenderPipeline:
    """Singleton instance of end-to-end TenderPipeline."""
    db = DatabaseManager(DEFAULT_DB_PATH)
    return TenderPipeline(config=DEFAULT_CONFIG, db_manager=db)


db_manager = get_database_manager()
pipeline = get_pipeline()
review_manager = ReviewManager(db_manager)
report_generator = ReportGenerator(db_manager)

# Manage user authentication in session state
if "authenticated_user" not in st.session_state:
    st.session_state["authenticated_user"] = None

user = st.session_state["authenticated_user"]


# =============================================================================
# LANDING PAGE & AUTHENTICATION (When user is not signed in)
# =============================================================================
if user is None:
    st.markdown(
        """
        <div class='tg-header' style='text-align: center; margin-top: 1rem; margin-bottom: 2rem;'>
            <div class='privacy-pill'>🔒 100% Local Inference &nbsp;•&nbsp; Zero Third-Party Transmission</div>
            <div style='font-size: 2.3rem; font-weight: 800; color: #0F172A; letter-spacing: -0.03em;'>
                🛡️ TenderGuard
            </div>
            <div style='font-size: 1.1rem; color: #475569; max-width: 650px; margin: 0.5rem auto 0 auto; line-height: 1.5;'>
                Intelligent Tender Risk Detection & Contradiction Analyzer
            </div>
            <div style='font-size: 0.88rem; color: #64748B; margin-top: 0.25rem;'>
                Transformer-Based Legal NLP Architecture for Public Procurement Auditing
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_hero, col_auth = st.columns([1.35, 1], gap="large")

    with col_hero:
        st.markdown(
            """
            <div class='tg-hero-card'>
                <div style='font-size: 1.15rem; font-weight: 700; color: #0F172A; margin-bottom: 0.5rem;'>
                    Automated Procurement Risk Discovery
                </div>
                <div style='font-size: 0.9rem; color: #475569; margin-bottom: 1.25rem; line-height: 1.5;'>
                    TenderGuard safeguards procurement officers and bidding agencies by detecting drafting inconsistencies, 
                    conflicting financial specifications, and chronological timeline collisions.
                </div>
                
                <div class='tg-feature-row'>
                    <div class='tg-feature-icon'>📄</div>
                    <div>
                        <div class='tg-feature-title'>PyMuPDF Extraction & Clause Segmentation</div>
                        <div class='tg-feature-desc'>Preserves exact page numbering, clause IDs, and character offsets while filtering headers and page artifacts.</div>
                    </div>
                </div>

                <div class='tg-feature-row'>
                    <div class='tg-feature-icon'>🤖</div>
                    <div>
                        <div class='tg-feature-title'>Transformer Zero-Shot Classification</div>
                        <div class='tg-feature-desc'>Categorizes clauses into 8 procurement domains (Financial, Technical, Deadlines, Legal, Penalties) using contextual embeddings.</div>
                    </div>
                </div>

                <div class='tg-feature-row'>
                    <div class='tg-feature-icon'>⚖️</div>
                    <div>
                        <div class='tg-feature-title'>Bidirectional NLI Contradiction Engine</div>
                        <div class='tg-feature-desc'>Pretrained sequence model (<code>distilbert-base-uncased-mnli</code>) evaluates cross-clause entailment and contradiction.</div>
                    </div>
                </div>

                <div class='tg-feature-row'>
                    <div class='tg-feature-icon'>🔢</div>
                    <div>
                        <div class='tg-feature-title'>Numerical & Milestone Consistency</div>
                        <div class='tg-feature-desc'>Dedicated rule engines extract currencies, percentages, and calendar milestones to identify conflicting terms.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_auth:
        st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
        auth_tab_login, auth_tab_signup = st.tabs(["Sign In", "Create Account"])

        with auth_tab_login:
            st.markdown("<div style='font-weight: 600; font-size: 1rem; margin-bottom: 0.85rem; color: #0F172A;'>Sign in to your account</div>", unsafe_allow_html=True)
            login_ident = st.text_input("Username or Email", placeholder="e.g. auditor or auditor@tenderguard.org", key="login_ident")
            login_pwd = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd")

            st.write("")
            if st.button("Sign In", type="primary", use_container_width=True):
                if not login_ident or not login_pwd:
                    st.error("Please enter both username/email and password.")
                else:
                    auth_user = db_manager.authenticate_user(login_ident, login_pwd)
                    if auth_user:
                        st.session_state["authenticated_user"] = auth_user
                        st.rerun()
                    else:
                        st.error("Invalid credentials. Please verify your username and password.")

            st.markdown("<div style='text-align: center; margin: 0.75rem 0; color: #94A3B8; font-size: 0.8rem;'>— OR —</div>", unsafe_allow_html=True)

            if st.button("⚡ Sign In with Demo Account", use_container_width=True):
                demo_user = db_manager.authenticate_user("auditor", "tenderguard2026")
                if demo_user:
                    st.session_state["authenticated_user"] = demo_user
                    st.rerun()

            st.caption("Default Demo Account: `auditor` / `tenderguard2026`")

        with auth_tab_signup:
            st.markdown("<div style='font-weight: 600; font-size: 1rem; margin-bottom: 0.85rem; color: #0F172A;'>Create Auditor Profile</div>", unsafe_allow_html=True)
            su_name = st.text_input("Full Name", placeholder="e.g. Jane Doe, Senior Auditor", key="su_name")
            su_email = st.text_input("Work Email", placeholder="e.g. j.doe@agency.gov", key="su_email")
            su_user = st.text_input("Choose Username", placeholder="e.g. jdoe_audit", key="su_user")
            su_org = st.text_input("Organization / Agency", placeholder="e.g. Ministry of Infrastructure", key="su_org")
            su_role = st.selectbox(
                "Role",
                ["Procurement Specialist", "Compliance Auditor", "Legal Counsel", "Contract Manager", "Bid Analyst"],
                key="su_role"
            )
            su_pwd = st.text_input("Password", type="password", placeholder="At least 6 characters", key="su_pwd")
            su_pwd_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pwd_conf")

            st.write("")
            if st.button("Register & Get Started", type="primary", use_container_width=True):
                if not su_name or not su_email or not su_user or not su_pwd:
                    st.error("Please fill in all required fields.")
                elif su_pwd != su_pwd_conf:
                    st.error("Passwords do not match.")
                elif len(su_pwd) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    try:
                        db_manager.create_user(
                            username=su_user,
                            email=su_email,
                            full_name=su_name,
                            password=su_pwd,
                            organization=su_org,
                            role=su_role
                        )
                        new_user = db_manager.authenticate_user(su_user, su_pwd)
                        st.session_state["authenticated_user"] = new_user
                        st.success("Account created successfully! Redirecting...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Registration failed: Username or email may already be in use. ({e})")

        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# AUTHENTICATED APPLICATION (When user is signed in)
# =============================================================================
else:
    # Sidebar User Profile & Sign Out
    st.sidebar.markdown(
        f"""
        <div class='tg-user-card'>
            <div class='tg-user-name'>👤 {user['full_name']}</div>
            <div class='tg-user-role'>{user.get('role', 'Auditor')}</div>
            <div class='tg-user-org'>{user.get('organization') or 'Public Authority'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Sign Out", use_container_width=True):
        st.session_state["authenticated_user"] = None
        st.session_state.pop("benchmark_results", None)
        st.rerun()

    st.sidebar.markdown("<div style='margin: 0.75rem 0;'></div>", unsafe_allow_html=True)

    nav_page = st.sidebar.radio(
        "Navigation",
        [
            "Home Dashboard",
            "Upload & Analyze",
            "Clause Explorer",
            "Findings Explorer",
            "Export Reports",
            "Model Evaluation & Benchmark",
            "Project Info & Limitations",
        ],
        index=0,
    )

    # Document selection in sidebar
    docs = db_manager.list_documents()
    doc_options = {d["id"]: f"{d['filename']} ({d['id']})" for d in docs}
    selected_doc_id = None

    if docs:
        st.sidebar.markdown("<div style='margin-top: 1.5rem; margin-bottom: 0.5rem; font-size: 0.8rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em;'>Active Tender</div>", unsafe_allow_html=True)
        selected_doc_id = st.sidebar.selectbox(
            "Select Document",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            index=0,
            label_visibility="collapsed"
        )
        if st.sidebar.button("🗑️ Delete Tender", type="secondary", use_container_width=True):
            db_manager.delete_document(selected_doc_id)
            st.sidebar.success(f"Document {selected_doc_id} deleted.")
            st.rerun()

    # -------------------------------------------------------------
    # 1. HOME DASHBOARD
    # -------------------------------------------------------------
    if nav_page == "Home Dashboard":
        st.markdown(
            f"""
            <div class='tg-header'>
                <div class='tg-title'>TenderGuard Overview</div>
                <div class='tg-subtitle'>Active Auditor: <b>{user['full_name']}</b> &nbsp;•&nbsp; {user.get('organization') or 'Procurement Division'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_findings = db_manager.get_findings_with_reviews()
        total_docs = len(docs)
        total_findings = len(all_findings)
        high_priority = sum(1 for f in all_findings if f.get("priority") == "High")
        pending_review = sum(1 for f in all_findings if f.get("review_status") == "Pending Review")

        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Analyzed Tenders", total_docs)
        col2.metric("Total Findings Flagged", total_findings)
        col3.metric("High-Priority Risks", high_priority, delta=f"{high_priority} critical" if high_priority else None, delta_color="inverse")
        col4.metric("Pending Human Reviews", pending_review)

        st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1.5, 1], gap="medium")

        with col_left:
            st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.75rem;'>Procurement Audit Workflow</div>", unsafe_allow_html=True)
            st.markdown(
                """
                - **Ingest & Extract**: Upload tender PDFs to parse text, detect scanned pages, and segment clauses with page preservation.
                - **Transformer Classification**: Clauses are classified into 8 domain categories without manual heuristics.
                - **Contradiction Discovery**: Bidirectional NLI and numerical/date engines identify conflicting requirements.
                - **Human Auditor Review**: Inspect side-by-side evidence, review diagnostic scores, and record audit determinations.
                - **Formal Reports**: Export structured audit records in PDF, CSV, and JSON formats.
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.75rem;'>Active NLP Engine</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div style='font-size: 0.85rem; color: #334155; line-height: 1.8;'>
                    • <b>Dense Classifier</b>: <code>{DEFAULT_CONFIG.models.classifier_model_name}</code><br/>
                    • <b>Embedding Model</b>: <code>{DEFAULT_CONFIG.models.embedding_model_name}</code><br/>
                    • <b>NLI Model</b>: <code>{DEFAULT_CONFIG.models.nli_model_name}</code><br/>
                    • <b>Similarity Threshold</b>: <code>{DEFAULT_CONFIG.models.similarity_threshold}</code><br/>
                    • <b>Contradiction Threshold</b>: <code>{DEFAULT_CONFIG.models.contradiction_threshold}</code><br/>
                    • <b>Privacy</b>: 100% Local Inference Active
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        if docs:
            st.markdown("<div style='font-size: 1.1rem; font-weight: 700; color: #0F172A; margin: 1.25rem 0 0.75rem 0;'>Analyzed Document Registry</div>", unsafe_allow_html=True)
            doc_df = pd.DataFrame(docs)[["id", "filename", "total_pages", "scanned_pages_count", "status", "uploaded_at"]]
            doc_df.columns = ["Document ID", "Filename", "Pages", "Scanned Pages", "Status", "Uploaded At"]
            st.dataframe(doc_df, use_container_width=True, hide_index=True)
        else:
            st.info("No documents analyzed yet. Go to **Upload & Analyze** to process your first tender.")

    # -------------------------------------------------------------
    # 2. UPLOAD & ANALYZE
    # -------------------------------------------------------------
    elif nav_page == "Upload & Analyze":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>Document Ingestion & Transformer Analysis</div>
                <div class='tg-subtitle'>Upload a tender specification PDF to execute real-time Transformer NLP inference.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sample_pdf_path = DATA_DIR / "sample_tender.pdf"
        has_sample = sample_pdf_path.is_file()

        upload_tab, demo_tab = st.tabs(["Upload Specification PDF", "Run Bundled Demo Tender"])

        target_source = None
        target_filename = None

        with upload_tab:
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Select Tender PDF File (Max 50MB)",
                type=["pdf"],
                help="PyMuPDF parses the document page-by-page, extracts clauses, and detects scanned pages."
            )
            if uploaded_file is not None:
                target_source = uploaded_file.read()
                target_filename = uploaded_file.name

        with demo_tab:
            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class='tg-card'>
                    <div style='font-weight: 600; font-size: 0.95rem; color: #0F172A; margin-bottom: 0.25rem;'>Pre-bundled Demo Specification: <code>sample_tender.pdf</code></div>
                    <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1rem; line-height: 1.5;'>
                        A realistic municipal IT infrastructure tender containing deliberate contradictions: deposit discrepancies ($50,000 vs $75,000), conflicting completion dates, and conflicting warranty terms.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("🚀 Analyze Demo Tender", type="primary", disabled=not has_sample):
                target_source = sample_pdf_path.read_bytes()
                target_filename = "sample_tender.pdf"

        if target_source and target_filename:
            st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
            st.markdown(f"#### Processing Document: `{target_filename}`")
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def update_progress(stage: str, pct: float, msg: str):
                progress_bar.progress(pct)
                status_text.markdown(f"**Stage: {stage} ({int(pct * 100)}%)** — {msg}")

            try:
                with st.spinner("Executing Transformer inference pipeline..."):
                    result = pipeline.process_pdf(
                        file_source=target_source,
                        filename=target_filename,
                        progress_callback=update_progress
                    )
                progress_bar.progress(1.0)
                status_text.markdown("✅ **Analysis Completed Successfully!**")

                st.success(
                    f"Document **{result['doc_id']}** analyzed in **{result['elapsed_seconds']}s**. "
                    f"Extracted {result['clauses_count']} clauses across {result['total_pages']} pages. "
                    f"Identified {result['findings_count']} potential risks."
                )

                # Summary metrics
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Pages", result["total_pages"])
                m2.metric("Clauses Extracted", result["clauses_count"])
                m3.metric("Candidate Pairs Evaluated", result["candidate_pairs_count"])
                m4.metric("Evidence-Backed Findings", result["findings_count"])

                st.info("Navigate to **Findings Explorer** to inspect side-by-side evidence and submit reviews.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

    # -------------------------------------------------------------
    # 3. CLAUSE EXPLORER
    # -------------------------------------------------------------
    elif nav_page == "Clause Explorer":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>Clause Explorer</div>
                <div class='tg-subtitle'>Browse, filter, and inspect segmented clauses with Transformer category predictions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.info("No tender documents analyzed yet. Please upload a document in **Upload & Analyze** first.")
        else:
            doc = db_manager.get_document(selected_doc_id)
            clauses = db_manager.get_clauses(selected_doc_id)

            st.caption(f"Active Document: **{doc['filename']}** (ID: `{selected_doc_id}` | Total Clauses: {len(clauses)})")

            # Filters
            c_filter1, c_filter2, c_filter3 = st.columns([1.5, 1, 2])
            with c_filter1:
                cat_filter = st.selectbox("Category", ["All Categories"] + TENDER_CATEGORIES)
            with c_filter2:
                pages_list = sorted(list({c["page_number"] for c in clauses}))
                page_filter = st.selectbox("Page", ["All Pages"] + [f"Page {p}" for p in pages_list])
            with c_filter3:
                search_query = st.text_input("Search Text", placeholder="Search keyword or clause number...")

            # Apply filters
            filtered_clauses = clauses
            if cat_filter != "All Categories":
                filtered_clauses = [c for c in filtered_clauses if c.get("category") == cat_filter]
            if page_filter != "All Pages":
                p_num = int(page_filter.replace("Page ", ""))
                filtered_clauses = [c for c in filtered_clauses if c["page_number"] == p_num]
            if search_query:
                q = search_query.lower()
                filtered_clauses = [
                    c for c in filtered_clauses
                    if q in c["original_text"].lower() or (c.get("clause_number") and q in c["clause_number"].lower())
                ]

            st.markdown(f"<div style='font-size: 0.88rem; color: #64748B; margin-bottom: 0.75rem;'>Showing <b>{len(filtered_clauses)}</b> of {len(clauses)} clauses</div>", unsafe_allow_html=True)

            for c in filtered_clauses:
                cat_name = c.get("category") or "Unclassified"
                conf = f"{c.get('category_confidence', 0.0):.1%}" if c.get('category_confidence') else "N/A"
                header_num = f" | No: {c['clause_number']}" if c.get("clause_number") else ""

                with st.expander(f"Clause {c['id']} — Page {c['page_number']}{header_num} [{cat_name}]"):
                    st.markdown(
                        f"<div style='margin-bottom: 0.5rem; font-size: 0.85rem; color: #475569;'>"
                        f"<b>Category:</b> <code>{cat_name}</code> (Confidence: {conf}) &nbsp;•&nbsp; "
                        f"<b>Span:</b> chars {c.get('char_start')}–{c.get('char_end')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"<div class='tg-clause-box-neutral'>{c['original_text']}</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 4. FINDINGS EXPLORER & HUMAN REVIEW
    # -------------------------------------------------------------
    elif nav_page == "Findings Explorer":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>Findings Explorer & Human Review</div>
                <div class='tg-subtitle'>Evidence-backed contradiction and risk analysis with human review decision workflow.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.info("No tender documents available. Please upload a tender document first.")
        else:
            all_f = db_manager.get_findings_with_reviews(doc_id=selected_doc_id)
            rev_summary = review_manager.get_review_summary(doc_id=selected_doc_id)

            # Metric summary row
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Findings", len(all_f))
            m2.metric("Confirmed Risks", rev_summary.get("Confirmed Risk", 0))
            m3.metric("Pending Review", rev_summary.get("Pending Review", 0))
            m4.metric("False Positives", rev_summary.get("False Positive", 0))

            st.markdown("<div style='margin: 1.25rem 0;'></div>", unsafe_allow_html=True)

            # Filter Bar
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                prio_filter = st.selectbox("Priority", ["All Priorities", "High", "Medium", "Low"])
            with f_col2:
                status_filter = st.selectbox("Review Status", ["All Statuses"] + REVIEW_STATUS_CHOICES)
            with f_col3:
                f_types = sorted(list({f["finding_type"] for f in all_f}))
                type_filter = st.selectbox("Finding Type", ["All Types"] + f_types)
            with f_col4:
                search_finding = st.text_input("Search Findings", placeholder="Search text...")

            filtered_findings = all_f
            if prio_filter != "All Priorities":
                filtered_findings = [f for f in filtered_findings if f["priority"] == prio_filter]
            if status_filter != "All Statuses":
                filtered_findings = [f for f in filtered_findings if f.get("review_status") == status_filter]
            if type_filter != "All Types":
                filtered_findings = [f for f in filtered_findings if f["finding_type"] == type_filter]
            if search_finding:
                sf = search_finding.lower()
                filtered_findings = [
                    f for f in filtered_findings
                    if sf in f["explanation"].lower() or sf in f["clause_a_text"].lower() or (f.get("clause_b_text") and sf in f["clause_b_text"].lower())
                ]

            st.markdown(f"<div style='font-size: 0.88rem; color: #64748B; margin-bottom: 0.75rem;'>Displaying <b>{len(filtered_findings)}</b> findings</div>", unsafe_allow_html=True)

            if not filtered_findings:
                st.info("No findings match the current filter selection.")

            for f in filtered_findings:
                status_val = f.get("review_status", "Pending Review")
                prio_badge = render_priority_badge(f["priority"])
                status_badge = render_status_badge(status_val)

                expander_title = (
                    f"[{f['priority']}] {f['id']} — {f['finding_type']} ({f['page_numbers']}) — {status_val}"
                )

                with st.expander(expander_title, expanded=(f["priority"] == "High")):
                    st.markdown(
                        f"""
                        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.85rem; flex-wrap: wrap;'>
                            {prio_badge}
                            {status_badge}
                            <span class='tg-badge' style='background-color: #F1F5F9; color: #334155;'>Type: {f['finding_type']}</span>
                            <span class='tg-badge' style='background-color: #F1F5F9; color: #334155;'>Pages: {f['page_numbers']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Explanation Card
                    st.markdown(
                        f"""
                        <div style='background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1rem; font-size: 0.9rem; color: #92400E;'>
                            <b>Potential Inconsistency:</b> {f['explanation']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Side-by-side Evidence
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"<div style='font-weight: 600; font-size: 0.88rem; color: #4F46E5; margin-bottom: 0.4rem;'>Clause {f['clause_a_id']} Evidence:</div>", unsafe_allow_html=True)
                        st.markdown(f"<div class='tg-clause-box-a'>{f['clause_a_text']}</div>", unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"<div style='font-weight: 600; font-size: 0.88rem; color: #D97706; margin-bottom: 0.4rem;'>Clause {f['clause_b_id'] or 'N/A'} Evidence:</div>", unsafe_allow_html=True)
                        b_text = f.get('clause_b_text') or '<i>Single-clause requirement</i>'
                        st.markdown(f"<div class='tg-clause-box-b'>{b_text}</div>", unsafe_allow_html=True)

                    # Model Diagnostics
                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                    sim_str = f"{f['semantic_similarity']:.2f}" if f.get('semantic_similarity') is not None else "N/A"
                    d_col1.metric("Semantic Similarity", sim_str)
                    contra_str = f"{f['nli_contradiction_score']:.1%}" if f.get('nli_contradiction_score') is not None else "N/A"
                    d_col2.metric("NLI Contradiction Risk", contra_str, help=f"Label: {f.get('nli_label', 'N/A')}")
                    d_col3.metric("Rule Triggered", f["rule_triggered"])
                    d_col4.metric("Inference Model", f.get("model_identifier", "N/A"))

                    # Human Review Decision Form
                    st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
                    st.markdown("<div style='font-weight: 700; font-size: 0.95rem; color: #0F172A; margin-bottom: 0.5rem;'>Auditor Review Decision</div>", unsafe_allow_html=True)

                    rev_col1, rev_col2 = st.columns([1, 2])
                    with rev_col1:
                        new_status = st.selectbox(
                            "Decision Status",
                            options=REVIEW_STATUS_CHOICES,
                            index=REVIEW_STATUS_CHOICES.index(status_val) if status_val in REVIEW_STATUS_CHOICES else 0,
                            key=f"status_{f['id']}"
                        )
                        reviewer = st.text_input(
                            "Reviewer Name",
                            value=f.get("reviewer_name") or user["full_name"],
                            key=f"name_{f['id']}"
                        )

                    with rev_col2:
                        comment = st.text_area(
                            "Audit Notes / Justification",
                            value=f.get("reviewer_comment") or "",
                            placeholder="Add procurement audit notes, clause amendment recommendations, or false-positive rationale...",
                            key=f"comment_{f['id']}",
                            height=98
                        )

                    if st.button("Save Review Decision", key=f"btn_{f['id']}", type="primary"):
                        review_manager.submit_review(
                            finding_id=f["id"],
                            status=new_status,
                            comment=comment,
                            reviewer_name=reviewer
                        )
                        st.success(f"Review saved for {f['id']}: {new_status}")
                        st.rerun()

    # -------------------------------------------------------------
    # 5. EXPORT REPORTS
    # -------------------------------------------------------------
    elif nav_page == "Export Reports":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>Export Formal Audit Reports</div>
                <div class='tg-subtitle'>Download procurement audit documentation in PDF, CSV, or JSON format.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.info("No tender documents available for export. Please upload a document first.")
        else:
            doc = db_manager.get_document(selected_doc_id)
            findings = db_manager.get_findings_with_reviews(doc_id=selected_doc_id)
            clauses = db_manager.get_clauses(selected_doc_id)

            st.caption(f"Active Document: **{doc['filename']}** (`{selected_doc_id}`) &nbsp;•&nbsp; {len(findings)} Findings &nbsp;•&nbsp; {len(clauses)} Clauses")

            col1, col2, col3 = st.columns(3, gap="medium")

            with col1:
                st.markdown(
                    """
                    <div class='tg-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📄</div>
                            <div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.35rem;'>PDF Audit Report</div>
                            <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1.25rem; line-height: 1.5;'>
                                Formal ReportLab document with executive summary, side-by-side evidence tables, and auditor sign-off.
                            </div>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    pdf_bytes = report_generator.export_pdf(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"TenderGuard_Report_{selected_doc_id}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col2:
                st.markdown(
                    """
                    <div class='tg-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📊</div>
                            <div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.35rem;'>CSV Findings Dataset</div>
                            <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1.25rem; line-height: 1.5;'>
                                Tabular CSV export containing all findings, clause pairs, similarity metrics, and auditor review decisions.
                            </div>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    csv_data = report_generator.export_csv(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_data,
                        file_name=f"TenderGuard_Findings_{selected_doc_id}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to generate CSV: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

            with col3:
                st.markdown(
                    """
                    <div class='tg-card' style='height: 100%; display: flex; flex-direction: column; justify-content: space-between;'>
                        <div>
                            <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🔌</div>
                            <div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.35rem;'>JSON Structured Data</div>
                            <div style='font-size: 0.85rem; color: #64748B; margin-bottom: 1.25rem; line-height: 1.5;'>
                                Full hierarchical JSON with document metadata, extracted clause spans, candidate pairs, and rule diagnostics.
                            </div>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                try:
                    json_data = report_generator.export_json(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download JSON File",
                        data=json_data,
                        file_name=f"TenderGuard_Data_{selected_doc_id}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"Failed to generate JSON: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 6. MODEL EVALUATION & BENCHMARK
    # -------------------------------------------------------------
    elif nav_page == "Model Evaluation & Benchmark":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>Model Evaluation & Comparative Benchmark</div>
                <div class='tg-subtitle'>Empirical benchmark evaluating Rule-based, Semantic Similarity, and Hybrid pipelines.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        benchmark_cases = load_benchmark_dataset()

        if not benchmark_cases:
            st.warning("No evaluation dataset available. Status: **Not evaluated**.")
        else:
            st.caption(f"Annotated Benchmark Suite: **{len(benchmark_cases)}** synthetic legal clause pairs.")

            evaluator = BenchmarkEvaluator()

            if st.button("Execute Comparative Benchmark Suite", type="primary"):
                with st.spinner("Running comparative evaluation across all 3 pipeline configurations..."):
                    metrics_list = evaluator.run_all_benchmarks(benchmark_cases)

                st.session_state["benchmark_results"] = metrics_list
                st.success("Benchmark completed successfully.")

            if "benchmark_results" in st.session_state:
                metrics_list = st.session_state["benchmark_results"]

                st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("#### Comparative Metrics Summary")
                summary_rows = []
                for m in metrics_list:
                    summary_rows.append({
                        "Pipeline Configuration": m.pipeline_name,
                        "Status": m.status,
                        "Test Cases": m.total_cases,
                        "Precision": f"{m.precision:.1%}",
                        "Recall": f"{m.recall:.1%}",
                        "F1-Score": f"{m.f1_score:.1%}",
                        "False Positive Rate": f"{m.fpr:.1%}",
                        "Avg Latency": f"{m.avg_processing_time_ms:.1f} ms",
                        "TP / FP / TN / FN": f"{m.confusion_matrix['TP']} / {m.confusion_matrix['FP']} / {m.confusion_matrix['TN']} / {m.confusion_matrix['FN']}"
                    })

                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.markdown("#### Confusion Matrices")
                cm_cols = st.columns(len(metrics_list))
                for col, m in zip(cm_cols, metrics_list):
                    with col:
                        st.markdown(f"**{m.pipeline_name}**")
                        cm = m.confusion_matrix
                        cm_df = pd.DataFrame(
                            [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]],
                            index=["Actual Negative (0)", "Actual Positive (1)"],
                            columns=["Pred Negative (0)", "Pred Positive (1)"]
                        )
                        st.dataframe(cm_df, use_container_width=True)

            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            st.markdown("#### Ground-Truth Test Cases Inspector")
            st.caption("Inspect the 6 synthetic test cases: numerical conflicts, date conflicts, complementary requirements, unrelated clauses, repeated clauses, and ambiguous requirements.")

            c_types = sorted(list({c.case_type for c in benchmark_cases}))
            case_type_filter = st.selectbox("Filter by Case Type", ["All Types"] + c_types)

            filtered_cases = benchmark_cases
            if case_type_filter != "All Types":
                filtered_cases = [c for c in filtered_cases if c.case_type == case_type_filter]

            for case in filtered_cases:
                label_badge = "<span class='tg-badge badge-high'>🔴 Conflict (1)</span>" if case.ground_truth_label == 1 else "<span class='tg-badge badge-confirmed'>🟢 Consistent (0)</span>"
                with st.expander(f"{case.id} [{case.case_type}]"):
                    st.markdown(f"{label_badge} &nbsp; <b>Description:</b> {case.description}", unsafe_allow_html=True)
                    col_ca, col_cb = st.columns(2)
                    with col_ca:
                        st.markdown(f"<div class='tg-clause-box-a'><b>Clause A:</b><br/>{case.clause_a}</div>", unsafe_allow_html=True)
                    with col_cb:
                        st.markdown(f"<div class='tg-clause-box-b'><b>Clause B:</b><br/>{case.clause_b}</div>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 7. PROJECT INFO & LIMITATIONS
    # -------------------------------------------------------------
    elif nav_page == "Project Info & Limitations":
        st.markdown(
            """
            <div class='tg-header'>
                <div class='tg-title'>System Architecture & Ethical Boundaries</div>
                <div class='tg-subtitle'>System documentation, Transformer model provenance, and legal disclaimers.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.75rem;'>Pipeline Flowchart</div>", unsafe_allow_html=True)
        st.markdown(
            """
            ```mermaid
            flowchart TD
                PDF[Tender PDF Upload] --> Extractor[PyMuPDF Page Extractor & Scanned Page Detection]
                Extractor --> Segmenter[Legal Clause Segmenter & Offset Preserver]
                Segmenter --> DB[(SQLite Database)]
                Segmenter --> Classifier[Transformer Zero-Shot Classifier]
                Classifier --> DB
                Segmenter --> ST[Sentence-BERT Dense Embedding Generator]
                ST --> Retrieval[Candidate Pair Semantic Retrieval]
                Retrieval --> NLI[Pretrained NLI Transformer Model]
                Retrieval --> NumEngine[Dedicated Numerical Inconsistency Engine]
                Retrieval --> DateEngine[Dedicated Date & Deadline Engine]
                NLI & NumEngine & DateEngine --> Hybrid[Hybrid Synthesis Engine]
                Hybrid --> Findings[Evidence-Backed Findings]
                Findings --> DB
                DB --> Streamlit[Streamlit Interactive Dashboard]
                DB --> HumanReview[Human Review Decision Workflow]
                DB --> Exports[PDF ReportLab, CSV, JSON Exports]
            ```
            """
        )
        st.markdown("</div>", unsafe_allow_html=True)

        col_prov, col_disc = st.columns([1.2, 1], gap="medium")

        with col_prov:
            st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.75rem;'>Model Provenance & Specs</div>", unsafe_allow_html=True)
            st.markdown(
                """
                - **Dense Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
                  - 22.7M parameters · 384-dimensional dense vectors
                  - Computes semantic cosine similarity across clauses and candidate pairs.
                - **Clause Category Classifier**: Dense Contextual Transformer Classifier
                  - Categorizes clauses into 8 tender domain categories without hardcoding.
                - **Natural Language Inference Model**: `typeform/distilbert-base-uncased-mnli`
                  - 66M parameters · Pretrained Multi-Genre NLI sequence classification
                  - Evaluates directional and bidirectional entailment, neutral, and contradiction probabilities.
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_disc:
            st.markdown("<div class='tg-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-weight: 700; font-size: 1.05rem; color: #0F172A; margin-bottom: 0.75rem;'>Ethical Boundaries & Disclaimers</div>", unsafe_allow_html=True)
            st.markdown(
                """
                - **Assistive Diagnostic Tool**: TenderGuard is an academic NLP assistive research tool designed to flag drafting discrepancies for human verification.
                - **Cautious Terminology**: Findings use cautious wording (*"Potential inconsistency"*, *"Requires human review"*). Model predictions never assert legal invalidity.
                - **Local Privacy**: All processing runs locally by default without transmitting tender texts to third-party endpoints.
                - **Human Oversight**: Model predictions are kept strictly separated from procurement officers' review determinations in SQLite.
                """
            )
            st.markdown("</div>", unsafe_allow_html=True)
