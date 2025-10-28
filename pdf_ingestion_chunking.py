import os
import re
import json
import glob
import time
import pdfplumber
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def detect_section(page, text, min_size_diff=2.0):
    """
    Try to extract the section title from page text using regex patterns or font size, weight.
    Handles 10-K, 10Q, Presentations, Investor PDFs and similar formats.
    """
    patterns = [
        r"(Item\s+\d+[A-Za-z]?.\s*[A-Z][^\n]+)",  # e.g. "Item 7. Management’s Discussion..."
        r"(ITEM\s+\d+[A-Z]?.\s*[A-Z][^\n]+)",     # uppercase variant
        r"(^[A-Z][A-Z\s]{10,})"                   # fallback for all-caps section titles
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            return m.group(1).strip()

    # For presentation decks/investor pdfs
    try:
        words = page.extract_words(extra_attrs=["size", "fontname"])
        if not words:
            return None

        from collections import defaultdict
        lines = defaultdict(list)

        # Group words by y-position
        for w in words:
            lines[round(w["top"], -1)].append(w)

        # Compute global average font size to compare against
        avg_font_size = sum(float(w["size"]) for w in words) / len(words)

        # Define line scoring function
        def score_line(line):
            avg_size = sum(float(w["size"]) for w in line) / len(line)
            bold_bonus = any("Bold" in w["fontname"] or "Black" in w["fontname"] for w in line)
            return avg_size + (3 if bold_bonus else 0), avg_size

        # Score all lines
        scored_lines = [(y, line, *score_line(line)) for y, line in lines.items()]
        if not scored_lines:
            return None

        # Pick the top scoring line
        _, best_line, best_score, best_size = max(scored_lines, key=lambda x: x[2])

        # If the line isn't significantly larger or bold, discard it
        if best_size < avg_font_size + min_size_diff:
            # (e.g. all text is 10pt, best line is 11pt → not a section)
            return None

        # Skip lines that look too long (to avoid full sentences)
        if len(best_line) > 20:
            return None

        section_title = " ".join(w["text"] for w in best_line)
        return section_title.strip() if section_title else None

    except Exception:
        return None


def parse_pdf_with_tables(pdf_path, report_name, year=None):
    text_docs, table_docs = [], []
    current_section = None  # rolling context

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            # --- Try to detect section header ---
            section = detect_section(page, text)
            if section:
                current_section = section  # update rolling section

            # --- Create text Document ---
            text_docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": i,
                        "report": report_name,
                        "year": year,
                        "section": current_section,
                        "type": "text"
                    }
                )
            )

            # --- Create table Documents ---
            for t in tables:
                cleaned_table = [
                    [cell if cell is not None else "" for cell in row]
                    for row in t
                ]
                table_text = "\n".join(["\t".join(row) for row in cleaned_table])
                table_docs.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "page": i,
                            "report": report_name,
                            "year": year,
                            "section": current_section,
                            "type": "table"
                        }
                    )
                )

    return text_docs, table_docs


def parse_pdf_cached(pdf_path, report_name, year=None):
    PARSED_DIR = "cache/parsed_pdfs"
    os.makedirs(PARSED_DIR, exist_ok=True)
    cache_file = os.path.join(PARSED_DIR, f"{report_name}.json")

    try:
        # --- If cached and file not modified, skip re-parsing ---
        # pdf_mtime = os.path.getmtime(pdf_path)
        if os.path.exists(cache_file):
            # cache_mtime = os.path.getmtime(cache_file)
            # if cache_mtime > pdf_mtime:
            print(f"Using cached parse for {report_name}\n")
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            docs = [Document(**d) for d in data]
            return docs

        # --- Otherwise, parse fresh ---
        print(f"Parsing {os.path.basename(pdf_path)} ...\n")
        text_docs, table_docs = parse_pdf_with_tables(pdf_path, report_name, year)
        docs = text_docs + table_docs

        # Save to cache
        serializable_docs = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, ensure_ascii=False, indent=2)

        return docs
    except Exception as e:
        print(f"Error parsing PDF {report_name}: {e}")
        return []


def process_pdf(pdf_path):
    report_name = os.path.basename(pdf_path).replace(".pdf", "")
    return parse_pdf_cached(pdf_path, report_name)


def process_all():
    # --- Measure time!! ----
    start_total = time.perf_counter()
    all_docs = []
    pdf_folder = 'data/'

    for pdf_path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        doc = process_pdf(pdf_path)
        all_docs.extend(doc)

    # with ThreadPoolExecutor(max_workers=4) as executor:
    #     results = list(executor.map(process_pdf, glob.glob(os.path.join(pdf_folder, "*.pdf"))))
    #     for docs in results:
    #         all_docs.extend(docs)

    elapsed_total = time.perf_counter() - start_total
    print(f"Loaded {len(all_docs)} parsed documents in {elapsed_total:.2f}s.")

    return all_docs


def chunk_documents(all_docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(all_docs)
