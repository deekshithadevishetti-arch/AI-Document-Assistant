from flask import Flask, render_template, request
from pathlib import Path
from werkzeug.utils import secure_filename
import ollama

from rag.document_loader import load_document
from rag.text_chunker import chunk_text
from rag.embeddings import generate_embeddings
from rag.vector_store import add_documents
from rag.search import search_documents


app = Flask(__name__)

# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
OLLAMA_MODEL = "llama3.2"


# --------------------------------------------------
# Check allowed file
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Main page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    message = ""
    answer = ""
    sources = []

    if request.method == "POST":

        action = request.form.get("action", "")

        # ==================================================
        # UPLOAD DOCUMENT
        # ==================================================

        if action == "upload":

            uploaded_file = request.files.get("document")

            if not uploaded_file or not uploaded_file.filename:

                message = "Please select a PDF, TXT, or DOCX file."

            elif not allowed_file(uploaded_file.filename):

                message = "Only PDF, TXT, and DOCX files are supported."

            else:

                filename = secure_filename(
                    uploaded_file.filename
                )

                file_path = DATA_DIR / filename

                try:

                    # Save uploaded file
                    uploaded_file.save(file_path)

                    # Read document
                    text = load_document(file_path)

                    if not text.strip():

                        message = (
                            "No readable text was found in the document."
                        )

                    else:

                        # Split text into chunks
                        chunks = chunk_text(text)

                        if not chunks:

                            message = (
                                "Could not create text chunks."
                            )

                        else:

                            # Generate embeddings
                            embeddings = generate_embeddings(chunks)

                            # Store chunks in ChromaDB
                            add_documents(
                                chunks,
                                embeddings,
                                source=filename
                            )

                            message = (
                                f"Successfully processed {filename}. "
                                f"Added {len(chunks)} text chunks."
                            )

                except Exception as exc:

                    message = (
                        f"Document processing failed: {exc}"
                    )

        # ==================================================
        # ASK QUESTION
        # ==================================================

        elif action == "ask":

            question = request.form.get(
                "question",
                ""
            ).strip()

            if not question:

                message = "Please enter a question."

            else:

                try:

                    # Search document database
                    results = search_documents(
                        question,
                        top_k=3
                    )

                    if not results:

                        message = (
                            "No relevant documents found. "
                            "Please upload a document first."
                        )

                    else:

                        # Build context
                        context = "\n\n".join(
                            result["text"]
                            for result in results
                        )

                        # Get sources
                        sources = list(
                            dict.fromkeys(
                                result["source"]
                                for result in results
                            )
                        )

                        # Ask Ollama
                        response = ollama.chat(
                            model=OLLAMA_MODEL,
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        "You are a document question "
                                        "answering assistant. "
                                        "Answer using only the provided "
                                        "document context. "
                                        "If the answer is not present "
                                        "in the context, clearly say "
                                        "that the information was not "
                                        "found in the document."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        "Document context:\n\n"
                                        + context
                                        + "\n\nQuestion:\n"
                                        + question
                                    )
                                }
                            ],
                            options={
                                "num_predict": 512
                            }
                        )

                        answer = response["message"]["content"]

                except Exception as exc:

                    message = (
                        f"Could not generate an answer: {exc}"
                    )

    return render_template(
        "index.html",
        message=message,
        answer=answer,
        sources=sources
    )


# --------------------------------------------------
# Start Flask
# --------------------------------------------------

if __name__ == "__main__":

    app.run(
        debug=False,
        use_reloader=False
    )