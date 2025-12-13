import asyncio
from playwright.async_api import async_playwright
import os

async def html_to_pdf(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # "file://" path requires absolute path
        abs_html_path = f"file://{os.path.abspath(html_path)}"
        await page.goto(abs_html_path, wait_until="networkidle")
        await page.pdf(path=pdf_path, format="A4", print_background=True, margin={"top": "1cm", "right": "1cm", "bottom": "1cm", "left": "1cm"})
        await browser.close()

if __name__ == "__main__":
    import sys
    html_file = "template.html"
    pdf_file = "template.pdf"
    
    print(f"Converting {html_file} to {pdf_file}...")
    asyncio.run(html_to_pdf(html_file, pdf_file))
    print("PDF generation complete.")
