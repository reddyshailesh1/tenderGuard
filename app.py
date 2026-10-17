"""TenderGuard — Intelligent Tender Risk Detection and Contradiction Analyzer.
Streamlit Web Application Interface with User Authentication and Landing Page.
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

# Custom Styling for polished dashboard UI
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4A5568;
        margin-bottom: 1.2rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #1A365D;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 1.15rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .feature-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .auth-card {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E1;
        border-radius: 10px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .priority-high {
        color: #E53E3E;
        font-weight: 700;
        background-color: #FFF5F5;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #FED7D7;
    }
    .priority-medium {
        color: #DD6B20;
        font-weight: 700;
        background-color: #FFFAF0;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #FEEBC8;
    }
    .priority-low {
        color: #3182CE;
        font-weight: 700;
        background-color: #EBF8FF;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #BEE3F8;
    }
    .status-confirmed {
        color: #2F855A;
        font-weight: 700;
        background-color: #F0FFF4;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #C6F6D5;
    }
    .status-pending {
        color: #C05621;
        font-weight: 600;
        background-color: #FFFAF0;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #FEEBC8;
    }
    .clause-box {
        background-color: #FFFFFF;
        border: 1px solid #CBD5E0;
        border-radius: 6px;
        padding: 12px;
        height: 100%;
    }
    .privacy-badge {
        display: inline-block;
        background-color: #EDF2F7;
        color: #2D3748;
        border: 1px solid #CBD5E0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        margin-bottom: 12px;
    }
    .user-pill {
        background-color: #F1F5F9;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
    st.markdown("<div class='hero-title'>🛡️ TenderGuard</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>"
        "Intelligent Tender Risk Detection and Contradiction Analyzer<br/>"
        "<i>Academic Transformer-Based Legal NLP Architecture for Public Procurement Auditing</i>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_hero, col_auth = st.columns([1.5, 1.2])

    with col_hero:
        st.markdown(
            """
            <div class='privacy-badge'>🔒 <b>Local Processing Active</b> &nbsp;•&nbsp; Zero External Data Transmission</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### 🏛️ Automated Procurement Risk Discovery
            TenderGuard protects contracting authorities and bidding organizations from drafting inconsistencies, 
            conflicting financial terms, and chronological scheduling impossibilities.
            """
        )

        st.markdown(
            """
            <div class='feature-card'>
                <b>📄 PyMuPDF Page Extraction & Clause Segmentation</b><br/>
                Preserves exact page numbering, section hierarchy, clause IDs, and character offsets while filtering running headers and page artifacts.
            </div>
            <div class='feature-card'>
                <b>🤖 Transformer Zero-Shot Domain Classification</b><br/>
                Categorizes clauses into 8 procurement domains (Financial, Technical, Deadlines, Legal, Penalties, Eligibility) without hardcoding.
            </div>
            <div class='feature-card'>
                <b>⚖️ Bidirectional NLI Contradiction Engine</b><br/>
                Pretrained sequence classification model (<code>typeform/distilbert-base-uncased-mnli</code>) evaluating logical entailment and contradiction.
            </div>
            <div class='feature-card'>
                <b>🔢 Dedicated Numerical & Milestone Date Engines</b><br/>
                Extracts currencies, percentages, and calendar dates to catch conflicting deposits ($50K vs $75K), warranty terms, and impossible deadlines.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_auth:
        st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
        auth_tab_login, auth_tab_signup = st.tabs(["🔑 Sign In", "📝 Create Account"])

        with auth_tab_login:
            st.markdown("#### Access TenderGuard")
            login_ident = st.text_input("Username or Email", placeholder="e.g. auditor or auditor@tenderguard.org", key="login_ident")
            login_pwd = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pwd")

            col_btn1, col_btn2 = st.columns([1, 1.3])
            with col_btn1:
                if st.button("Sign In", type="primary", use_container_width=True):
                    if not login_ident or not login_pwd:
                        st.error("Please enter both username/email and password.")
                    else:
                        auth_user = db_manager.authenticate_user(login_ident, login_pwd)
                        if auth_user:
                            st.session_state["authenticated_user"] = auth_user
                            st.success(f"Welcome back, {auth_user['full_name']}!")
                            st.rerun()
                        else:
                            st.error("Invalid credentials. Please verify your username and password.")

            with col_btn2:
                if st.button("⚡ Quick Demo Sign In", use_container_width=True):
                    demo_user = db_manager.authenticate_user("auditor", "tenderguard2026")
                    if demo_user:
                        st.session_state["authenticated_user"] = demo_user
                        st.rerun()

            st.caption("Default Demo Account: `auditor` / `tenderguard2026`")

        with auth_tab_signup:
            st.markdown("#### New User Registration")
            su_name = st.text_input("Full Name", placeholder="e.g. Jane Doe, Senior Procurement Officer", key="su_name")
            su_email = st.text_input("Work Email", placeholder="e.g. j.doe@agency.gov", key="su_email")
            su_user = st.text_input("Choose Username", placeholder="e.g. jdoe_audit", key="su_user")
            su_org = st.text_input("Organization / Agency", placeholder="e.g. Department of Infrastructure", key="su_org")
            su_role = st.selectbox(
                "Role",
                ["Procurement Specialist", "Compliance Auditor", "Legal Counsel", "Contract Manager", "Bid Analyst"],
                key="su_role"
            )
            su_pwd = st.text_input("Password", type="password", placeholder="Create secure password", key="su_pwd")
            su_pwd_conf = st.text_input("Confirm Password", type="password", placeholder="Repeat password", key="su_pwd_conf")

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
                        # Automatically authenticate
                        new_user = db_manager.authenticate_user(su_user, su_pwd)
                        st.session_state["authenticated_user"] = new_user
                        st.success("Account created successfully! Redirecting...")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Registration failed: Username or email may already be registered. ({e})")

        st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# AUTHENTICATED APPLICATION (When user is signed in)
# =============================================================================
else:
    # Sidebar User Profile & Sign Out
    st.sidebar.markdown(
        f"""
        <div class='user-pill'>
            <b>👤 {user['full_name']}</b><br/>
            <small>@{user['username']} • {user.get('role', 'Auditor')}</small><br/>
            <small style='color: #64748B;'>{user.get('organization') or 'Public Authority'}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("🚪 Sign Out", use_container_width=True):
        st.session_state["authenticated_user"] = None
        st.session_state.pop("benchmark_results", None)
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    nav_page = st.sidebar.radio(
        "Modules",
        [
            "Home Dashboard",
            "Upload & Analyze",
            "Clause Explorer",
            "Findings Explorer",
            "Export Reports",
            "Model Evaluation & Benchmark",
            "Project Info & Limitations",
        ],
        index=1,  # Default to Upload & Analyze after sign in
    )

    # Document selection helper in sidebar
    docs = db_manager.list_documents()
    doc_options = {d["id"]: f"{d['filename']} ({d['id']})" for d in docs}
    selected_doc_id = None

    if docs:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 Active Tender")
        selected_doc_id = st.sidebar.selectbox(
            "Select Analyzed Document",
            options=list(doc_options.keys()),
            format_func=lambda x: doc_options[x],
            index=0,
        )
        if st.sidebar.button("🗑️ Delete Tender", type="secondary"):
            db_manager.delete_document(selected_doc_id)
            st.sidebar.success(f"Document {selected_doc_id} deleted.")
            st.rerun()

    # -------------------------------------------------------------
    # 1. HOME DASHBOARD
    # -------------------------------------------------------------
    if nav_page == "Home Dashboard":
        st.markdown("<div class='main-header'>🛡️ TenderGuard Overview</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='sub-header'>Logged in as <b>{user['full_name']}</b> ({user.get('organization') or 'Procurement Division'})</div>",
            unsafe_allow_html=True,
        )

        all_findings = db_manager.get_findings_with_reviews()
        total_docs = len(docs)
        total_findings = len(all_findings)
        high_priority = sum(1 for f in all_findings if f.get("priority") == "High")
        pending_review = sum(1 for f in all_findings if f.get("review_status") == "Pending Review")
        confirmed_risks = sum(1 for f in all_findings if f.get("review_status") == "Confirmed Risk")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Analyzed Tenders", total_docs)
        col2.metric("Total Findings", total_findings)
        col3.metric("High Priority Risks", high_priority, delta=f"{high_priority} critical", delta_color="inverse")
        col4.metric("Pending Human Review", pending_review)
        col5.metric("Confirmed Risks", confirmed_risks)

        st.markdown("---")

        col_left, col_right = st.columns([1.6, 1])

        with col_left:
            st.markdown("### 📋 Procurement Review Workflow")
            st.markdown(
                """
                1. **Upload Tender PDF**: Go to **Upload & Analyze** to process new tender documents.
                2. **Clause Categorization**: Review clauses segmented by page and classified into 8 procurement domains.
                3. **Inspect Contradictions**: Review side-by-side evidence in **Findings Explorer**, view NLI confidence scores, and record audit decisions.
                4. **Download Formal Reports**: Export audit reports in **PDF**, **CSV**, or **JSON** format.
                5. **Benchmark Integrity**: Test baseline models against our hybrid Transformer pipeline in **Model Evaluation**.
                """
            )

        with col_right:
            st.markdown("### ⚙️ Active NLP Engine")
            st.info(
                f"""
                - **Classifier**: `{DEFAULT_CONFIG.models.classifier_model_name}`
                - **Dense Retriever**: `{DEFAULT_CONFIG.models.embedding_model_name}`
                - **NLI Model**: `{DEFAULT_CONFIG.models.nli_model_name}`
                - **Similarity Threshold**: `{DEFAULT_CONFIG.models.similarity_threshold}`
                - **Contradiction Threshold**: `{DEFAULT_CONFIG.models.contradiction_threshold}`
                - **Storage**: SQLite (`tenderguard.db`)
                - **Security**: 100% Local Processing Active
                """
            )

        if docs:
            st.markdown("### 📑 Analyzed Documents Registry")
            doc_df = pd.DataFrame(docs)[["id", "filename", "total_pages", "scanned_pages_count", "status", "uploaded_at"]]
            doc_df.columns = ["Document ID", "Filename", "Pages", "Scanned Pages", "Status", "Uploaded At"]
            st.dataframe(doc_df, use_container_width=True)

    # -------------------------------------------------------------
    # 2. UPLOAD & ANALYZE
    # -------------------------------------------------------------
    elif nav_page == "Upload & Analyze":
        st.markdown("<div class='main-header'>📤 Tender Document Ingestion & Analysis</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>Upload a tender specification PDF to execute real-time Transformer NLP inference.</div>",
            unsafe_allow_html=True,
        )

        sample_pdf_path = DATA_DIR / "sample_tender.pdf"
        has_sample = sample_pdf_path.is_file()

        col_upload, col_demo = st.columns([2, 1])

        with col_upload:
            uploaded_file = st.file_uploader(
                "Select Tender PDF File (Max 50MB)",
                type=["pdf"],
                help="PyMuPDF parses the document page-by-page, extracts clauses, and detects scanned pages."
            )

        with col_demo:
            st.markdown("#### Or Run Bundled Demo Tender")
            st.caption("A multi-page tender specification containing realistic numerical, date, and warranty conflicts.")
            load_demo = st.button("🚀 Analyze Demo Tender (sample_tender.pdf)", disabled=not has_sample)

        target_source = None
        target_filename = None

        if uploaded_file is not None:
            target_source = uploaded_file.read()
            target_filename = uploaded_file.name
        elif load_demo and has_sample:
            target_source = sample_pdf_path.read_bytes()
            target_filename = "sample_tender.pdf"

        if target_source and target_filename:
            st.markdown("---")
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

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Pages", result["total_pages"])
                m2.metric("Clauses Extracted", result["clauses_count"])
                m3.metric("Candidate Pairs Evaluated", result["candidate_pairs_count"])
                m4.metric("Evidence-Backed Findings", result["findings_count"])

                st.info("💡 Navigate to **Findings Explorer** to inspect side-by-side evidence and submit reviews.")

            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.exception(e)

    # -------------------------------------------------------------
    # 3. CLAUSE EXPLORER
    # -------------------------------------------------------------
    elif nav_page == "Clause Explorer":
        st.markdown("<div class='main-header'>📑 Clause Explorer</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>Browse, filter, and inspect segmented clauses with Transformer category predictions.</div>",
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.warning("No tender documents analyzed yet. Please upload a document in 'Upload & Analyze' first.")
        else:
            doc = db_manager.get_document(selected_doc_id)
            clauses = db_manager.get_clauses(selected_doc_id)

            st.caption(f"Showing clauses for: **{doc['filename']}** (ID: `{selected_doc_id}` | Total Clauses: {len(clauses)})")

            # Filters
            c_filter1, c_filter2, c_filter3 = st.columns([1.5, 1, 2])
            with c_filter1:
                cat_filter = st.selectbox("Filter by Category", ["All Categories"] + TENDER_CATEGORIES)
            with c_filter2:
                pages_list = sorted(list({c["page_number"] for c in clauses}))
                page_filter = st.selectbox("Filter by Page", ["All Pages"] + [f"Page {p}" for p in pages_list])
            with c_filter3:
                search_query = st.text_input("Search Clause Text", placeholder="Search keyword or clause number...")

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

            st.markdown(f"**Found {len(filtered_clauses)} matching clauses**")

            for c in filtered_clauses:
                cat_name = c.get("category") or "Unclassified"
                conf = f"{c.get('category_confidence', 0.0):.1%}" if c.get('category_confidence') else "N/A"
                header_num = f" | Number: `{c['clause_number']}`" if c.get("clause_number") else ""

                with st.expander(f"Clause `{c['id']}` — Page {c['page_number']}{header_num} [{cat_name}]"):
                    st.markdown(f"**Category:** `{cat_name}` (Confidence: {conf})")
                    st.markdown(f"<div class='clause-box'>{c['original_text']}</div>", unsafe_allow_html=True)
                    st.caption(f"Offsets: chars {c.get('char_start')} to {c.get('char_end')}")

    # -------------------------------------------------------------
    # 4. FINDINGS EXPLORER & HUMAN REVIEW
    # -------------------------------------------------------------
    elif nav_page == "Findings Explorer":
        st.markdown("<div class='main-header'>🔍 Findings Explorer & Human Review</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>Evidence-backed contradiction and risk analysis with human review decision workflow.</div>",
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.warning("No tender documents available. Please upload a tender document first.")
        else:
            # Filters
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                prio_filter = st.selectbox("Priority", ["All Priorities", "High", "Medium", "Low"])
            with f_col2:
                status_filter = st.selectbox("Review Status", ["All Statuses"] + REVIEW_STATUS_CHOICES)
            with f_col3:
                all_f = db_manager.get_findings_with_reviews(doc_id=selected_doc_id)
                f_types = sorted(list({f["finding_type"] for f in all_f}))
                type_filter = st.selectbox("Finding Type", ["All Types"] + f_types)
            with f_col4:
                search_finding = st.text_input("Search Evidence / Explanations", placeholder="Filter by text...")

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

            # Metric summary row for active document
            rev_summary = review_manager.get_review_summary(doc_id=selected_doc_id)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Findings", len(all_f))
            m2.metric("Confirmed Risks", rev_summary.get("Confirmed Risk", 0))
            m3.metric("Pending Review", rev_summary.get("Pending Review", 0))
            m4.metric("False Positives", rev_summary.get("False Positive", 0))

            st.markdown(f"### Displaying {len(filtered_findings)} Findings")

            if not filtered_findings:
                st.info("No findings match the current filter selection.")

            for f in filtered_findings:
                prio_class = f"priority-{f['priority'].lower()}"
                status_val = f.get("review_status", "Pending Review")
                status_class = "status-confirmed" if status_val == "Confirmed Risk" else "status-pending"

                expander_title = (
                    f"[{f['priority']}] {f['id']} — {f['finding_type']} ({f['page_numbers']}) | Status: {status_val}"
                )

                with st.expander(expander_title, expanded=(f["priority"] == "High")):
                    # Top metadata line
                    st.markdown(
                        f"<span class='{prio_class}'>Priority: {f['priority']}</span> &nbsp; "
                        f"<span class='{status_class}'>Status: {status_val}</span> &nbsp; "
                        f"<b>Type:</b> {f['finding_type']} | <b>Locations:</b> {f['page_numbers']}",
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")

                    # Side-by-side evidence
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f"**Clause {f['clause_a_id']} Evidence:**")
                        st.markdown(f"<div class='clause-box'>{f['clause_a_text']}</div>", unsafe_allow_html=True)

                    with col_b:
                        st.markdown(f"**Clause {f['clause_b_id'] or 'N/A'} Evidence:**")
                        st.markdown(
                            f"<div class='clause-box'>{f.get('clause_b_text') or '<i>No paired clause</i>'}</div>",
                            unsafe_allow_html=True,
                        )

                    # Diagnostic details
                    st.markdown("#### Model Diagnostics & Triggered Rules")
                    d_col1, d_col2, d_col3 = st.columns(3)
                    sim_str = f"{f['semantic_similarity']:.2f}" if f.get('semantic_similarity') is not None else "N/A"
                    d_col1.metric("Semantic Similarity", sim_str)
                    contra_str = f"{f['nli_contradiction_score']:.1%}" if f.get('nli_contradiction_score') is not None else "N/A"
                    d_col2.metric("NLI Contradiction Risk", contra_str, help=f"Label: {f.get('nli_label', 'N/A')}")
                    d_col3.metric("Rule ID", f["rule_triggered"])

                    st.markdown(f"**Explanation:** {f['explanation']}")
                    st.caption(f"Inference Model: `{f['model_identifier']}`")

                    # Human review decision form
                    st.markdown("#### Human Review Decision")
                    rev_col1, rev_col2, rev_col3 = st.columns([1.5, 2.5, 1])

                    with rev_col1:
                        new_status = st.selectbox(
                            "Decision Status",
                            options=REVIEW_STATUS_CHOICES,
                            index=REVIEW_STATUS_CHOICES.index(status_val) if status_val in REVIEW_STATUS_CHOICES else 0,
                            key=f"status_{f['id']}"
                        )
                        reviewer = st.text_input(
                            "Reviewer Name / Title",
                            value=f.get("reviewer_name") or user["full_name"],
                            key=f"name_{f['id']}"
                        )

                    with rev_col2:
                        comment = st.text_area(
                            "Reviewer Comments & Justification",
                            value=f.get("reviewer_comment") or "",
                            placeholder="Add procurement audit notes, clause amendment recommendations, or false-positive rationale...",
                            key=f"comment_{f['id']}",
                            height=100
                        )

                    with rev_col3:
                        st.write("")
                        st.write("")
                        if st.button("💾 Save Review", key=f"btn_{f['id']}", type="primary"):
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
        st.markdown("<div class='main-header'>📥 Export Audit Reports</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>Download formal tender audit documentation in PDF, CSV, or JSON format.</div>",
            unsafe_allow_html=True,
        )

        if not selected_doc_id:
            st.warning("No tender documents available for export. Please upload a document first.")
        else:
            doc = db_manager.get_document(selected_doc_id)
            findings = db_manager.get_findings_with_reviews(doc_id=selected_doc_id)
            clauses = db_manager.get_clauses(selected_doc_id)

            st.info(f"Preparing report for **{doc['filename']}** (`{selected_doc_id}`) | {len(findings)} Findings | {len(clauses)} Clauses")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 📄 PDF Audit Report")
                st.write("Professional ReportLab document containing cover summary, findings breakdown, side-by-side evidence tables, and reviewer signatures.")
                try:
                    pdf_bytes = report_generator.export_pdf(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"TenderGuard_Report_{selected_doc_id}.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")

            with col2:
                st.markdown("### 📊 CSV Data Dump")
                st.write("Tabular CSV export containing all findings, clause IDs, similarity metrics, NLI probabilities, rules triggered, and review decisions.")
                try:
                    csv_data = report_generator.export_csv(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download CSV File",
                        data=csv_data,
                        file_name=f"TenderGuard_Findings_{selected_doc_id}.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Failed to generate CSV: {e}")

            with col3:
                st.markdown("### 🔌 JSON Machine-Readable")
                st.write("Complete structured JSON export containing document metadata, extracted clauses with character spans, candidate pairs, and findings.")
                try:
                    json_data = report_generator.export_json(selected_doc_id)
                    st.download_button(
                        label="⬇️ Download JSON File",
                        data=json_data,
                        file_name=f"TenderGuard_Data_{selected_doc_id}.json",
                        mime="application/json"
                    )
                except Exception as e:
                    st.error(f"Failed to generate JSON: {e}")

    # -------------------------------------------------------------
    # 6. MODEL EVALUATION & BENCHMARK
    # -------------------------------------------------------------
    elif nav_page == "Model Evaluation & Benchmark":
        st.markdown("<div class='main-header'>📊 Model Evaluation & Comparative Benchmark</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>Rigorous empirical benchmark evaluating Rule-based, Semantic Similarity, and Hybrid pipelines.</div>",
            unsafe_allow_html=True,
        )

        benchmark_cases = load_benchmark_dataset()

        if not benchmark_cases:
            st.warning("No evaluation dataset available. Status: **Not evaluated**.")
        else:
            st.markdown(f"**Benchmark Dataset:** `{len(benchmark_cases)}` manually annotated synthetic clause pairs.")

            evaluator = BenchmarkEvaluator()

            if st.button("▶️ Execute Benchmark Suite", type="primary"):
                with st.spinner("Running comparative evaluation across all 3 pipeline configurations..."):
                    metrics_list = evaluator.run_all_benchmarks(benchmark_cases)

                st.session_state["benchmark_results"] = metrics_list
                st.success("Benchmark completed!")

            if "benchmark_results" in st.session_state:
                metrics_list = st.session_state["benchmark_results"]

                # Comparison table
                st.markdown("### 📈 Comparative Metrics Summary")
                summary_rows = []
                for m in metrics_list:
                    summary_rows.append({
                        "Pipeline Approach": m.pipeline_name,
                        "Status": m.status,
                        "Total Cases": m.total_cases,
                        "Precision": f"{m.precision:.2%}",
                        "Recall": f"{m.recall:.2%}",
                        "F1-Score": f"{m.f1_score:.2%}",
                        "False Positive Rate": f"{m.fpr:.2%}",
                        "Avg Processing Time": f"{m.avg_processing_time_ms:.1f} ms",
                        "TP / FP / TN / FN": f"{m.confusion_matrix['TP']} / {m.confusion_matrix['FP']} / {m.confusion_matrix['TN']} / {m.confusion_matrix['FN']}"
                    })

                summary_df = pd.DataFrame(summary_rows)
                st.table(summary_df)

                # Confusion matrices
                st.markdown("### 🔲 Confusion Matrices")
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
                        st.dataframe(cm_df)

            st.markdown("---")
            st.markdown("### 📑 Ground-Truth Benchmark Cases Viewer")
            st.caption("Inspect the 6 required synthetic test cases: numerical conflicts, date conflicts, complementary requirements, unrelated clauses, repeated clauses, and ambiguous requirements.")

            c_types = sorted(list({c.case_type for c in benchmark_cases}))
            case_type_filter = st.selectbox("Filter Case Type", ["All Types"] + c_types)

            filtered_cases = benchmark_cases
            if case_type_filter != "All Types":
                filtered_cases = [c for c in filtered_cases if c.case_type == case_type_filter]

            for case in filtered_cases:
                label_tag = "🔴 Conflict / Contradiction (1)" if case.ground_truth_label == 1 else "🟢 Consistent / Non-conflict (0)"
                with st.expander(f"{case.id} [{case.case_type}] — {label_tag}"):
                    st.markdown(f"**Description:** {case.description}")
                    col_ca, col_cb = st.columns(2)
                    with col_ca:
                        st.markdown("**Clause A:**")
                        st.info(case.clause_a)
                    with col_cb:
                        st.markdown("**Clause B:**")
                        st.info(case.clause_b)

    # -------------------------------------------------------------
    # 7. PROJECT INFO & LIMITATIONS
    # -------------------------------------------------------------
    elif nav_page == "Project Info & Limitations":
        st.markdown("<div class='main-header'>ℹ️ Project Architecture & Limitations</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sub-header'>System documentation, Transformer model provenance, and legal disclaimers.</div>",
            unsafe_allow_html=True,
        )

        st.markdown("### 🏗️ Architectural Overview")
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

        st.markdown("### 🤖 Pretrained Models & Provenance")
        st.markdown(
            """
            1. **Dense Semantic Embedding Model**:
               - **Model**: `sentence-transformers/all-MiniLM-L6-v2`
               - **Parameters**: 22.7M parameters
               - **Output Dimensionality**: 384-dimensional dense vectors
               - **Role**: Computes cosine similarity across clauses and retrieves cross-section candidate pairs.
            2. **Clause Classifier**:
               - **Model**: Dense Contextual Transformer Prototype Classifier on `all-MiniLM-L6-v2`
               - **Role**: Categorizes clauses into 8 tender domain categories without arbitrary hardcoding.
            3. **Natural Language Inference Model**:
               - **Model**: `typeform/distilbert-base-uncased-mnli`
               - **Parameters**: 66M parameters
               - **Role**: Computes directional and bidirectional entailment, neutral, and contradiction probabilities on candidate pairs.
            """
        )

        st.markdown("### ⚠️ Ethical Boundaries & Legal Disclaimers")
        st.warning(
            """
            - **Assistive Diagnostic Tool**: TenderGuard is an academic NLP assistive research tool designed to flag drafting discrepancies.
            - **No Legal Invalidation**: The application uses cautious wording (*"Potential inconsistency"*, *"Requires human review"*). Model predictions never assert legal invalidity.
            - **Local Privacy**: All processing runs locally by default without transmitting tender texts to third-party endpoints.
            - **Human Oversight**: Model predictions are kept strictly separated from procurement officers' review determinations in SQLite.
            """
        )
