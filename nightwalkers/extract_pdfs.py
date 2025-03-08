import os
import json
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# Set Tesseract OCR path (MacOS users should check with `which tesseract`)
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# Input & Output folders
PDF_FOLDER = "pdfs/"
OUTPUT_FOLDER = "processed_pdfs/"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def extract_text(pdf_path):
    """Extract text per page using pdfplumber."""
    text_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text_data.append({
                "page": i + 1,
                "text": text.strip()
            })
    return text_data

def extract_tables(pdf_path):
    """Extract tables per page and format them correctly."""
    table_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            formatted_tables = []
            for table in tables:
                if not table:
                    continue
                headers = table[0] if table else []
                rows = table[1:] if len(table) > 1 else []
                formatted_tables.append({
                    "headers": headers,
                    "rows": rows
                })
            table_data.append({
                "page": i + 1,
                "tables": formatted_tables
            })
    return table_data

def extract_images_with_ocr(pdf_path, filename):
    """Extract images per page, apply OCR, and handle errors."""
    image_data = []
    doc = fitz.open(pdf_path)
    temp_image_paths = []  # Store image paths for cleanup

    for i, page in enumerate(doc):
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            img_path = os.path.join(OUTPUT_FOLDER, f"temp_img_{i+1}_{img_index}.png")

            # Save image temporarily
            with open(img_path, "wb") as img_file:
                img_file.write(image_bytes)

            try:
                # Apply OCR
                ocr_text = pytesseract.image_to_string(img_path).strip()
                image_data.append({
                    "page": i + 1,
                    "figures": {"ocr_text": ocr_text}
                })
            except pytesseract.TesseractError as e:
                print(f"⚠️ Skipping corrupted image on Page {i+1} in {filename}: {e}")
                image_data.append({
                    "page": i + 1,
                    "figures": {"ocr_text": ""}
                })

            # Store for cleanup
            temp_image_paths.append(img_path)

    # Cleanup: Remove temporary images
    for img_path in temp_image_paths:
        try:
            os.remove(img_path)
        except Exception as e:
            print(f"Warning: Could not delete {img_path}. Error: {e}")

    return image_data

def process_pdf(pdf_file):
    """Process a PDF and generate structured JSON output."""
    pdf_path = os.path.join(PDF_FOLDER, pdf_file)
    output_path = os.path.join(OUTPUT_FOLDER, f"{pdf_file}.json")

    # Check if the PDF was already processed
    if os.path.exists(output_path):
        print(f"✅ Skipping {pdf_file}, already processed.")
        return

    text_data = extract_text(pdf_path)
    table_data = extract_tables(pdf_path)
    image_data = extract_images_with_ocr(pdf_path, pdf_file)

    pages = []
    for i in range(len(text_data)):  # Ensure we align data per page
        page_number = text_data[i]["page"]

        page_entry = {
            "filename": pdf_file,
            "page": page_number,
            "text": text_data[i]["text"],
            "tables": table_data[i]["tables"] if i < len(table_data) else [],
            "figures": image_data[i]["figures"] if i < len(image_data) else {"ocr_text": ""}
        }
        pages.append(page_entry)

    # Save extracted data
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, indent=4, ensure_ascii=False)

    print(f"✅ Processed: {pdf_file}")

# Run extraction on all PDFs in the folder
for pdf_file in os.listdir(PDF_FOLDER):
    if pdf_file.endswith(".pdf"):
        process_pdf(pdf_file)
