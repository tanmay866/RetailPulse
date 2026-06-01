"""Generate a self-contained HTML EDA report and a summary JSON from the EDA notebook."""
import json
import os

import nbformat
import pandas as pd
from nbconvert import HTMLExporter

NOTEBOOK = 'notebooks/eda.ipynb'
OUT_HTML = 'reports/eda_report.html'
OUT_JSON = 'reports/eda_summary.json'


def _patch_css(html: str) -> str:
    """Add standard CSS properties alongside vendor-prefixed ones to silence linter warnings."""
    html = html.replace(
        '-webkit-appearance: none;\n}',
        '-webkit-appearance: none;\n  appearance: none;\n}',
    )
    html = html.replace(
        '-webkit-print-color-adjust: exact;',
        '-webkit-print-color-adjust: exact;\n  print-color-adjust: exact;',
    )
    return html


def generate_html_report():
    os.makedirs('reports', exist_ok=True)
    with open(NOTEBOOK, encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    exporter = HTMLExporter()
    exporter.embed_images = True
    html_body, _ = exporter.from_notebook_node(nb)
    html_body = _patch_css(html_body)
    with open(OUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html_body)
    print(f'saved {OUT_HTML}')


def generate_summary_json():
    clean = pd.read_csv('data/processed/retail_clean.csv', low_memory=False)
    rfm   = pd.read_csv('data/processed/rfm_scores.csv')

    clean['InvoiceDate'] = pd.to_datetime(clean['InvoiceDate'])
    clean['Revenue']     = clean['Quantity'] * clean['Price']

    summary = {
        'cleaned_rows':      int(len(clean)),
        'date_range': {
            'start': str(clean['InvoiceDate'].min().date()),
            'end':   str(clean['InvoiceDate'].max().date()),
        },
        'total_revenue_gbp': round(float(clean['Revenue'].sum()), 2),
        'unique_customers':  int(clean['Customer ID'].nunique()),
        'unique_products':   int(clean['Description'].nunique()),
        'rfm_segment_counts': (
            rfm['Segment'].value_counts().to_dict()
            if 'Segment' in rfm.columns else {}
        ),
        'top_10_products_by_revenue': (
            clean.groupby('Description')['Revenue'].sum()
            .nlargest(10).round(2).to_dict()
        ),
    }

    os.makedirs('reports', exist_ok=True)
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'saved {OUT_JSON}')


if __name__ == '__main__':
    generate_html_report()
    generate_summary_json()
