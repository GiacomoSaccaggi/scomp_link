# -*- coding: utf-8 -*-
"""
██████╗ ██████╗ ███████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝██║  ██║█████╗  
██╔═══╝ ██║  ██║██╔══╝  
██║     ██████╔╝██║     
╚═╝     ╚═════╝ ╚═╝     

PDF conversion utilities for scomp-link.
Provides functions to convert Markdown and HTML files to PDF format.

Dependencies (optional):
    - markdown: for Markdown to HTML conversion
    - weasyprint: for HTML to PDF rendering
"""

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

try:
    from weasyprint import HTML
    HAS_WEASYPRINT = True
except (ImportError, OSError):
    HAS_WEASYPRINT = False


def markdown_to_pdf(input_path: str, output_path: str = None, css: str = None) -> str:
    """
    Convert a Markdown file to PDF.

    Parameters
    ----------
    input_path : str
        Path to the .md file.
    output_path : str, optional
        Path for the output PDF. Defaults to input_path with .pdf extension.
    css : str, optional
        Custom CSS string to style the PDF.

    Returns
    -------
    str
        Path to the generated PDF file.
    """
    if not HAS_MARKDOWN:
        raise ImportError("markdown package not installed. Install with: pip install markdown")
    if not HAS_WEASYPRINT:
        raise ImportError("weasyprint package not installed. Install with: pip install weasyprint")

    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.pdf'

    with open(input_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'toc'])

    full_html = _wrap_html(html_content, css)

    HTML(string=full_html).write_pdf(output_path)

    return output_path


def html_to_pdf(input_path: str, output_path: str = None, css: str = None) -> str:
    """
    Convert an HTML file to PDF.

    Parameters
    ----------
    input_path : str
        Path to the .html file.
    output_path : str, optional
        Path for the output PDF. Defaults to input_path with .pdf extension.
    css : str, optional
        Additional CSS string to inject into the HTML before rendering.

    Returns
    -------
    str
        Path to the generated PDF file.
    """
    if not HAS_WEASYPRINT:
        raise ImportError("weasyprint package not installed. Install with: pip install weasyprint")

    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '.pdf'

    if css:
        with open(input_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        style_tag = f'<style>{css}</style>'
        if '</head>' in html_content:
            html_content = html_content.replace('</head>', f'{style_tag}</head>')
        else:
            html_content = style_tag + html_content
        HTML(string=html_content).write_pdf(output_path)
    else:
        HTML(filename=input_path).write_pdf(output_path)

    return output_path


from scomp_link.utils.colors import MAIN, MAIN_DARK


def _wrap_html(body: str, css: str = None) -> str:
    """Wrap HTML body content in a full HTML document with optional CSS."""
    default_css = f"""
        body {{ font-family: sans-serif; font-size: 14px; line-height: 1.6; margin: 40px; color: #333; }}
        h1, h2, h3 {{ color: {MAIN_DARK}; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 12px; border-radius: 4px; overflow-x: auto; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: {MAIN}; color: white; }}
    """
    style = css if css else default_css
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>{style}</style></head>
<body>{body}</body>
</html>"""
