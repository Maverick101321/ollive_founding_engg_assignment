import json
import os
from collections import defaultdict
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib import colors
from reportlab.lib.units import inch

def generate_report():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    scores_path = os.path.join(base_dir, 'scores.json')
    if not os.path.exists(scores_path):
        print(f"Error: {scores_path} not found.")
        return
        
    with open(scores_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    category_scores = defaultdict(lambda: {'qwen': [], 'gemini': []})
    
    for item in data:
        cat = item.get('category', 'unknown')
        if 'qwen_score' in item:
            category_scores[cat]['qwen'].append(item['qwen_score'])
        if 'gemini_score' in item:
            category_scores[cat]['gemini'].append(item['gemini_score'])
            
    # Calculate averages
    averages = []
    for cat, scores in category_scores.items():
        q_avg = sum(scores['qwen']) / len(scores['qwen']) if scores['qwen'] else 0
        g_avg = sum(scores['gemini']) / len(scores['gemini']) if scores['gemini'] else 0
        averages.append([cat.capitalize(), f"{q_avg:.2f}", f"{g_avg:.2f}"])
        
    pdf_path = os.path.join(base_dir, 'eval_report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    
    title_style = styles['Title']
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.gray,
        alignment=1, # Center
        spaceAfter=20
    )
    
    elements = []
    
    # Title & Subtitle
    elements.append(Paragraph("AI Assistant Evaluation Report", title_style))
    elements.append(Paragraph("Qwen2.5-7B-Instruct vs Gemini — Judged by Llama-3-70B", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Summary Table
    table_data = [['Category', 'Qwen2.5-7B Avg Score', 'Gemini Avg Score']] + averages
    t = Table(table_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 20))
    
    # Charts
    chart1_path = os.path.join(base_dir, 'charts', 'bar_chart.png')
    chart2_path = os.path.join(base_dir, 'charts', 'refusal_rate.png')
    chart3_path = os.path.join(base_dir, 'charts', 'safety_breakdown.png')
    
    # Two charts side by side
    img1 = Image(chart1_path, width=3*inch, height=2.25*inch) if os.path.exists(chart1_path) else Paragraph("bar_chart.png missing", styles['Normal'])
    img2 = Image(chart2_path, width=3*inch, height=2.25*inch) if os.path.exists(chart2_path) else Paragraph("refusal_rate.png missing", styles['Normal'])
    
    chart_table = Table([[img1, img2]], colWidths=[3.2*inch, 3.2*inch])
    chart_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'), 
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    elements.append(chart_table)
    elements.append(Spacer(1, 20))
    
    # Full width chart
    if os.path.exists(chart3_path):
        img3 = Image(chart3_path, width=6*inch, height=3*inch)
        img3.hAlign = 'CENTER'
        elements.append(img3)
    else:
        elements.append(Paragraph("safety_breakdown.png missing", styles['Normal']))
        
    elements.append(Spacer(1, 20))
    
    # Recommendations
    elements.append(Paragraph("<b>Recommendations:</b>", styles['Heading3']))
    rec_text = (
        "1. Consider using Gemini for tasks requiring stricter safety and refusal compliance.<br/>"
        "2. Qwen2.5-7B can be favored for general factual tasks if less restrictive alignment is desired.<br/>"
        "3. Monitor safety breakdowns regularly to adjust system prompts dynamically."
    )
    elements.append(Paragraph(rec_text, styles['Normal']))
    
    doc.build(elements)
    print(f"Report generated successfully at {pdf_path}")

if __name__ == "__main__":
    generate_report()
