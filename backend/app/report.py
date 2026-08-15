from __future__ import annotations

import csv
import io
import json


def build_json_report(investigation_id: str, result: dict) -> bytes:
    return json.dumps({"investigation_id": investigation_id, **result}, indent=2, default=str).encode()


def build_csv_report(result: dict) -> bytes:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Section", "Key", "Value"])

    for k, v in result.get("evaluation", {}).items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            writer.writerow(["evaluation", k, v])

    for i, f in enumerate(result.get("findings", [])):
        writer.writerow(["finding", f"{i+1}. {f['title']}", f"severity={f['severity']}; {f['interpretation']}"])

    for s in result.get("slices", []):
        writer.writerow(["slice", s["rule"], f"error_rate={s['group_error_rate']:.3f} n={s['sample_size']} gap={s['performance_gap']:.3f}"])

    if result.get("drift"):
        for d in result["drift"]:
            writer.writerow(["drift", d["feature"], f"psi={d['psi']:.3f} status={d['status']}"])

    return buf.getvalue().encode()


def build_pdf_report(investigation_id: str, result: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], textColor=colors.HexColor("#1e1b4b"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], textColor=colors.HexColor("#4c1d95"))
    body = styles["BodyText"]

    story = [
        Paragraph("MODELLENS Investigation Report", title_style),
        Paragraph(f"Investigation ID: {investigation_id}", body),
        Spacer(1, 12),
    ]

    mi = result.get("model_info", {})
    di = result.get("dataset_info", {})
    story.append(Paragraph("Model & Dataset", h2))
    story.append(
        Table(
            [
                ["Task type", mi.get("task_type", "-")],
                ["Estimator", mi.get("estimator_class", "-")],
                ["Samples evaluated", str(di.get("n_samples", "-"))],
                ["Features used", str(di.get("n_features_used", "-"))],
                ["Target column", str(di.get("target_column", "-"))],
            ],
            colWidths=[180, 300],
        )
    )
    story.append(Spacer(1, 16))

    story.append(Paragraph("Evaluation Metrics", h2))
    ev = result.get("evaluation", {})
    rows = [[k, str(round(v, 4)) if isinstance(v, float) else str(v)]
            for k, v in ev.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
    t = Table(rows, colWidths=[220, 260])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
    story.append(t)
    story.append(Spacer(1, 16))

    story.append(Paragraph("Investigation Findings", h2))
    for f in result.get("findings", []):
        story.append(Paragraph(f"[{f['severity'].upper()}] {f['title']}", styles["Heading4"]))
        story.append(Paragraph(f["interpretation"], body))
        story.append(Paragraph(f"Recommendation: {f['recommendation']}", body))
        story.append(Spacer(1, 8))

    if result.get("slices"):
        story.append(PageBreak())
        story.append(Paragraph("Discovered Problematic Slices", h2))
        rows = [["Rule", "Sample size", "Group error rate", "Gap"]]
        for s in result["slices"][:10]:
            rows.append([s["rule"][:60], str(s["sample_size"]), f"{s['group_error_rate']:.3f}", f"{s['performance_gap']:.3f}"])
        t = Table(rows, colWidths=[260, 70, 90, 60])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.append(t)

    if result.get("drift"):
        story.append(Spacer(1, 16))
        story.append(Paragraph("Data Drift", h2))
        rows = [["Feature", "PSI", "Status"]]
        for d in result["drift"]:
            rows.append([d["feature"], f"{d['psi']:.3f}" if d["psi"] == d["psi"] else "n/a", d["status"]])
        t = Table(rows, colWidths=[220, 100, 120])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 8)]))
        story.append(t)

    story.append(Spacer(1, 20))
    story.append(Paragraph(
        "Limitations: This report reflects statistical associations observed on the evaluated dataset. "
        "It does not establish causation, does not guarantee real-world significance, and does not certify "
        "the model for any particular use case.", styles["Italic"]
    ))

    doc.build(story)
    return buf.getvalue()
