"""Load contract text from an uploaded file."""

import io


def load_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="replace")


def load_pdf(file_bytes: bytes) -> str:
    # Imported lazily so a broken/missing pypdf install (or its optional
    # crypto backend) can't break .txt-only usage, which is the primary
    # path for this prototype.
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def load_contract(file_bytes: bytes, filename: str) -> str:
    """Dispatch on file extension. Raises ValueError for unsupported types."""
    name = filename.lower()
    if name.endswith(".txt"):
        return load_txt(file_bytes)
    if name.endswith(".pdf"):
        return load_pdf(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}. Use .txt or .pdf.")
