import os
import json

# Input & Output folders
PROCESSED_FOLDER = "processed_pdfs/"
CLEANUP_FOLDER = "cleanup_pdfs/"

os.makedirs(CLEANUP_FOLDER, exist_ok=True)

# Set a minimum text length to keep a page
TEXT_LENGTH_THRESHOLD = 30  # Adjust this if needed

def clean_json(filename):
    """Load, filter, and save cleaned JSON files."""
    input_path = os.path.join(PROCESSED_FOLDER, filename)
    output_path = os.path.join(CLEANUP_FOLDER, filename)

    # Load JSON
    with open(input_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    # Filter out pages that meet the removal criteria
    cleaned_pages = [
        page for page in pages
        if len(page["text"].strip()) > TEXT_LENGTH_THRESHOLD  # Keep if text is long enough
        or page["tables"]  # Keep if there are tables
        or page["figures"].get("ocr_text", "").strip()  # Keep if OCR text is present
    ]

    # Save cleaned JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_pages, f, indent=4, ensure_ascii=False)

    print(f"✅ Cleaned: {filename} | Pages before: {len(pages)} | Pages after: {len(cleaned_pages)}")

# Process all JSON files in the folder
for json_file in os.listdir(PROCESSED_FOLDER):
    if json_file.endswith(".json"):
        clean_json(json_file)
