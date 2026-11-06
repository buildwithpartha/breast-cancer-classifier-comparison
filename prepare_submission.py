"""Generate a printable submission with real links and a real lab screenshot.

Usage: python prepare_submission.py --repo URL --app URL --screenshot PATH
Open submission.html in a browser and print to PDF, then verify its links.
"""
import argparse
import base64
from html import escape
import mimetypes
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--app', required=True)
    parser.add_argument('--screenshot', required=True, type=Path)
    args = parser.parse_args()
    for url in [args.repo, args.app]:
        if not url.startswith('https://'):
            parser.error('Links must be real HTTPS URLs.')
    mime = mimetypes.guess_type(args.screenshot.name)[0]
    if mime not in ['image/png', 'image/jpeg']:
        parser.error('Screenshot must be PNG or JPEG.')
    import markdown
    image = base64.b64encode(args.screenshot.read_bytes()).decode()
    readme = markdown.markdown((ROOT / 'README.md').read_text(encoding='utf-8'), extensions=['tables', 'fenced_code'])
    content = f'''<!doctype html><html><head><meta charset="utf-8"><title>ML Assignment 2</title>
    <style>body{{font:14px Arial;max-width:1000px;margin:40px auto;line-height:1.5}}
    table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{border:1px solid #aaa;padding:6px}}
    img{{max-width:100%}}pre{{white-space:pre-wrap}}a{{overflow-wrap:anywhere}}
    @media print{{.readme{{break-before:page}}tr{{break-inside:avoid}}}}</style></head><body>
    <h1>Machine Learning — Assignment 2</h1><h2>1. GitHub repository</h2>
    <a href="{escape(args.repo, quote=True)}">{escape(args.repo)}</a>
    <h2>2. Live Streamlit app</h2><a href="{escape(args.app, quote=True)}">{escape(args.app)}</a>
    <h2>3. BITS Virtual Lab execution screenshot</h2><img alt="Actual BITS Lab execution" src="data:{mime};base64,{image}">
    <section class="readme"><h2>4. GitHub README content</h2>{readme}</section></body></html>'''
    output = ROOT / 'submission.html'
    output.write_text(content, encoding='utf-8')
    print('Created', output, '— print to PDF and check all links before submitting.')


if __name__ == '__main__':
    main()
