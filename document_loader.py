import os

from pypdf import PdfReader
from docx import Document


def load_pdf(filepath):
    reader = PdfReader(filepath)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def load_docx(filepath):
    document = Document(filepath)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    return text


def load_document(filepath):
    extension = os.path.splitext(filepath)[1].lower()

    if extension == ".pdf":
        return load_pdf(filepath)

    elif extension == ".txt":
        return load_txt(filepath)

    elif extension == ".docx":
        return load_docx(filepath)

    else:
        raise ValueError(f"Unsupported file type: {extension}")