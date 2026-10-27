
import hashlib
import re
import json
from pathlib import Path

import pymupdf


# ============================================================
# 1. CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Supports headings such as:
# References
# 7. References
# VII. REFERENCES
# References and Notes
REFERENCE_HEADING = re.compile(
    r"(?im)^[ \t]*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)[.)]?[ \t]+)?"
    r"(?:references|bibliography|works cited|literature cited)"
    r"(?:[ \t]+and[ \t]+notes)?[ \t.:]*$"
)

# Matches both "Abstract" and "Abstract—Large Language Models..."
ABSTRACT_HEADING = re.compile(
    r"(?im)^[ \t]*abstract\b"
)


# ============================================================
# 2. TEXT CLEANING
# ============================================================

def clean_text(text):
    """Normalize whitespace and PDF line-break artifacts."""

    if not text:
        return ""

    text = text.replace("\x00", "")

    # Join words split by a line-ending hyphen.
    # This is a heuristic and may affect genuinely hyphenated terms.
    text = re.sub(
        r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)",
        "",
        text
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def trim_front_matter(text):
    """
    Remove author and affiliation text preceding an identifiable
    abstract heading. Preserve the original text if none is found.
    """

    match = ABSTRACT_HEADING.search(text)

    if match:
        return text[match.start():].strip()

    return text


# ============================================================
# 3. PDF EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_bytes):
    """Extract PDF text page by page and exclude references."""

    pages = []
    reached_references = False

    with pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    ) as pdf:

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            if reached_references:
                break

            text = clean_text(page.get_text("text"))

            if not text:
                continue

            # Detect a standalone bibliography heading.
            match = REFERENCE_HEADING.search(text)

            if match:
                text = text[:match.start()].strip()
                reached_references = True

                print(
                    f"Reference section detected on "
                    f"page {page_number}."
                )

            # Trim front matter only on the first page.
            if page_number == 1:
                text = trim_front_matter(text)

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages


# ============================================================
# 4. CHUNKING
# ============================================================

def split_into_chunks(text):
    """Split text into overlapping character chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


# ============================================================
# 5. PROCESS PDF FILES
# ============================================================

def process_pdfs(pdf_files):
    """
    Accept a list of (filename, PDF bytes) tuples.

    Return chunks with unique IDs and source metadata.
    """

    all_chunks = []

    for filename, pdf_bytes in pdf_files:
        source = Path(filename).name

        file_hash = hashlib.sha256(
            pdf_bytes
        ).hexdigest()[:12]

        pages = extract_pdf_pages(pdf_bytes)

        if not pages:
            print(
                f"Warning: no selectable text found in {source}"
            )
            continue

        source_chunk_count = 0

        for page_data in pages:
            page_number = page_data["page"]

            page_chunks = split_into_chunks(
                page_data["text"]
            )

            for chunk_index, chunk_text in enumerate(page_chunks):
                chunk_id = (
                    f"{file_hash}_page_{page_number}"
                    f"_chunk_{chunk_index}"
                )

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page_number,
                        "chunk_id": chunk_id
                    }
                })

                source_chunk_count += 1

        print(
            f"{source}: {len(pages)} pages processed, "
            f"{source_chunk_count} chunks created."
        )

    if not all_chunks:
        raise ValueError(
            "No text could be extracted. "
            "Check whether the PDFs contain selectable text."
        )

    ids = [chunk["id"] for chunk in all_chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    return all_chunks


# ============================================================
# 6. PROCESS THE DATA FOLDER
# ============================================================

def process_pdf_folder(folder="data"):
    """Process all PDF files inside the specified folder."""

    folder_path = Path(folder)

    if not folder_path.exists():
        raise FileNotFoundError(
            f"Folder not found: {folder_path}"
        )

    pdf_files = [
        (path.name, path.read_bytes())
        for path in sorted(folder_path.glob("*.pdf"))
    ]

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {folder_path}"
        )

    print(f"Found {len(pdf_files)} PDF files.")

    return process_pdfs(pdf_files)


# ============================================================
# 7. SAVE CHUNKS
# ============================================================

def save_chunks(
    chunks,
    output_path="output/chunks.json"
):
    """Save processed chunks as JSON."""

    output_file = Path(output_path)
    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_file.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Saved {len(chunks)} chunks to {output_file}")


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    chunks = process_pdf_folder("data")

    save_chunks(chunks)

    print("\n" + "=" * 50)
    print("INGESTION COMPLETED")
    print("=" * 50)
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk preview:")
        print(chunks[0])
