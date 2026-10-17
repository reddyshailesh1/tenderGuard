"""Multi-format report generation (PDF, CSV, JSON) for TenderGuard."""
import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from tenderguard.db.session import DatabaseManager


class ReportGenerator:
    """Generates professional PDF, CSV, and JSON audit reports from database records."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    # ------------------ CSV Export ------------------

    def export_csv(self, doc_id: str) -> str:
        """Generate flat CSV representation of all findings and human review decisions."""
        findings = self.db.get_findings_with_reviews(doc_id=doc_id)
        output = io.StringIO()
        fieldnames = [
            "finding_id", "doc_id", "finding_type", "priority", "page_numbers",
            "clause_a_id", "clause_a_text", "clause_b_id", "clause_b_text",
            "semantic_similarity", "nli_label", "nli_contradiction_score",
            "rule_triggered", "explanation", "model_identifier",
            "review_status", "reviewer_name", "reviewer_comment", "reviewed_at"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for f in findings:
            writer.writerow({
                "finding_id": f.get("id"),
                "doc_id": f.get("doc_id"),
                "finding_type": f.get("finding_type"),
                "priority": f.get("priority"),
                "page_numbers": f.get("page_numbers"),
                "clause_a_id": f.get("clause_a_id"),
                "clause_a_text": f.get("clause_a_text"),
                "clause_b_id": f.get("clause_b_id") or "",
                "clause_b_text": f.get("clause_b_text") or "",
                "semantic_similarity": f.get("semantic_similarity") if f.get("semantic_similarity") is not None else "",
                "nli_label": f.get("nli_label") or "",
                "nli_contradiction_score": f.get("nli_contradiction_score") if f.get("nli_contradiction_score") is not None else "",
                "rule_triggered": f.get("rule_triggered"),
                "explanation": f.get("explanation"),
                "model_identifier": f.get("model_identifier"),
                "review_status": f.get("review_status", "Pending Review"),
                "reviewer_name": f.get("reviewer_name", ""),
                "reviewer_comment": f.get("reviewer_comment", ""),
                "reviewed_at": f.get("reviewed_at", "")
            })

        return output.getvalue()

    # ------------------ JSON Export ------------------

    def export_json(self, doc_id: str) -> str:
        """Generate structured JSON representation of document, clauses, and findings."""
        doc = self.db.get_document(doc_id)
        clauses = self.db.get_clauses(doc_id)
        findings = self.db.get_findings_with_reviews(doc_id=doc_id)

        report = {
            "application": "TenderGuard — Intelligent Tender Risk Detection and Contradiction Analyzer",
            "export_timestamp": datetime.now().isoformat(),
            "document": doc,
            "clauses_count": len(clauses),
            "clauses": clauses,
            "findings_count": len(findings),
            "findings": findings
        }
        return json.dumps(report, indent=2, default=str)

    # ------------------ PDF Export (ReportLab) ------------------

    def export_pdf(self, doc_id: str) -> bytes:
        """Generate a formatted, evidence-backed PDF report using ReportLab."""
        doc = self.db.get_document(doc_id)
        if not doc:
            raise ValueError(f"Document '{doc_id}' not found.")

        findings = self.db.get_findings_with_reviews(doc_id=doc_id)
        clauses = self.db.get_clauses(doc_id)

        buffer = io.BytesIO()
        pdf = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Custom typography styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#1A365D"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#4A5568"),
            spaceAfter=15
        )
        section_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#2B6CB0"),
            spaceBefore=12,
            spaceAfter=6
        )
        body_style = ParagraphStyle(
            "ReportBody",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#2D3748")
        )
        bold_style = ParagraphStyle(
            "ReportBold",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            fontName="Helvetica-Bold",
            textColor=colors.HexColor("#1A202C")
        )
        box_style = ParagraphStyle(
            "ClauseBox",
            parent=styles["Normal"],
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#2D3748")
        )

        elements: List[Any] = []

        # 1. Header Banner
        elements.append(Paragraph("TenderGuard Audit Report", title_style))
        elements.append(Paragraph(
            f"Automated Tender Risk & Contradiction Analysis | Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=12))

        # 2. Executive Summary Metadata Table
        high_cnt = sum(1 for f in findings if f.get("priority") == "High")
        med_cnt = sum(1 for f in findings if f.get("priority") == "Medium")
        low_cnt = sum(1 for f in findings if f.get("priority") == "Low")
        confirmed_cnt = sum(1 for f in findings if f.get("review_status") == "Confirmed Risk")

        meta_data = [
            [Paragraph("Document ID:", bold_style), Paragraph(doc["id"], body_style),
             Paragraph("Total Pages:", bold_style), Paragraph(str(doc["total_pages"]), body_style)],
            [Paragraph("File Name:", bold_style), Paragraph(doc["filename"], body_style),
             Paragraph("Scanned Pages:", bold_style), Paragraph(str(doc.get("scanned_pages_count", 0)), body_style)],
            [Paragraph("Total Clauses:", bold_style), Paragraph(str(len(clauses)), body_style),
             Paragraph("Total Findings:", bold_style), Paragraph(f"{len(findings)} ({high_cnt} High, {med_cnt} Med)", body_style)],
            [Paragraph("Analysis Status:", bold_style), Paragraph(doc["status"].capitalize(), body_style),
             Paragraph("Confirmed Risks:", bold_style), Paragraph(str(confirmed_cnt), body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[85, 185, 85, 185])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EDF2F7")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 14))

        # 3. Findings Section
        elements.append(Paragraph(f"Detailed Analysis Findings ({len(findings)} Detected)", section_style))

        if not findings:
            elements.append(Paragraph("No inconsistencies or contradictions detected in this tender document.", body_style))
        else:
            for idx, f in enumerate(findings):
                priority_color = "#E53E3E" if f["priority"] == "High" else ("#DD6B20" if f["priority"] == "Medium" else "#3182CE")
                status_color = "#38A169" if f.get("review_status") == "Confirmed Risk" else "#4A5568"

                finding_header_text = (
                    f"<b>Finding {f['id']}</b> | <b>Type:</b> {f['finding_type']} | "
                    f"<b>Priority:</b> <font color='{priority_color}'>{f['priority']}</font> | "
                    f"<b>Review Status:</b> <font color='{status_color}'>{f.get('review_status', 'Pending Review')}</font>"
                )

                # Clause evidence cells
                clause_a_box = Paragraph(f"<b>Clause {f['clause_a_id']}</b><br/>{f['clause_a_text']}", box_style)
                clause_b_box = Paragraph(
                    f"<b>Clause {f['clause_b_id'] or 'N/A'}</b><br/>{f.get('clause_b_text') or 'No paired clause'}",
                    box_style
                )

                # Scores & rule
                contra_score = f"{f['nli_contradiction_score']:.1%}" if f.get('nli_contradiction_score') is not None else "N/A"
                sim_score = f"{f['semantic_similarity']:.2f}" if f.get('semantic_similarity') is not None else "N/A"

                tech_details = (
                    f"<b>Locations:</b> {f['page_numbers']} | <b>Rule:</b> {f['rule_triggered']}<br/>"
                    f"<b>Semantic Similarity:</b> {sim_score} | <b>NLI Contradiction:</b> {contra_score} ({f.get('nli_label', 'N/A')})<br/>"
                    f"<b>Model:</b> {f['model_identifier']}"
                )

                reviewer_note = (
                    f"<b>Human Reviewer Note ({f.get('reviewer_name', 'None')}):</b> "
                    f"{f.get('reviewer_comment') or 'No comment provided.'}"
                )

                f_table_data = [
                    [Paragraph(finding_header_text, body_style)],
                    [Table([[clause_a_box, clause_b_box]], colWidths=[265, 265])],
                    [Paragraph(f"<b>Explanation:</b> {f['explanation']}", body_style)],
                    [Paragraph(tech_details, body_style)],
                    [Paragraph(reviewer_note, body_style)]
                ]

                f_table = Table(f_table_data, colWidths=[540])
                f_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F7FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E0")),
                    ("LINEBELOW", (0, 0), (-1, 0), 1, colors.HexColor("#E2E8F0")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]))

                elements.append(KeepTogether(f_table))
                elements.append(Spacer(1, 10))

        # 4. Disclaimer Footer
        elements.append(Spacer(1, 14))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E0"), spaceAfter=6))
        disclaimer = (
            "<b>Disclaimer:</b> TenderGuard is an academic NLP assistive diagnostic tool. "
            "Findings highlight potential drafting ambiguities and inconsistencies requiring human professional review. "
            "Model predictions do not constitute legal advice or formal determination of invalidity."
        )
        elements.append(Paragraph(disclaimer, ParagraphStyle("Disc", parent=styles["Normal"], fontSize=7, leading=9, textColor=colors.HexColor("#718096"))))

        pdf.build(elements)
        buffer.seek(0)
        return buffer.getvalue()
