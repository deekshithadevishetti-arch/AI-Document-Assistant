import streamlit as st
from pathlib import Path
from transformers import pipeline

from document_loader import load_document
from text_chunker import chunk_text
from embeddings import generate_embeddings
from vector_store import add_documents
from search import search_documents


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="AI Documentation Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Documentation Assistant")
st.write("Upload a document and ask questions about its content.")


# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(exist_ok=True)


# ==================================================
# AI QUESTION ANSWERING MODEL
# ==================================================

@st.cache_resource
def load_qa_model():

    return pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad"
    )


# ==================================================
# DOCUMENT UPLOAD
# ==================================================

st.header("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF, TXT, or DOCX file",
    type=["pdf", "txt", "docx"]
)


if uploaded_file is not None:

    filename = Path(uploaded_file.name).name

    file_path = DATA_DIR / filename

    if st.button("Process Document"):

        try:

            # Save uploaded document
            with open(file_path, "wb") as file:
                file.write(uploaded_file.getbuffer())

            # Read document
            text = load_document(file_path)

            if not text.strip():

                st.error(
                    "No readable text was found in the document."
                )

            else:

                # Create text chunks
                chunks = chunk_text(text)

                if not chunks:

                    st.error(
                        "Could not create text chunks."
                    )

                else:

                    # Generate embeddings
                    embeddings = generate_embeddings(chunks)

                    # Store documents
                    add_documents(
                        chunks,
                        embeddings,
                        source=filename
                    )

                    st.success(
                        f"Successfully processed {filename} "
                        f"with {len(chunks)} text chunks."
                    )

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# ==================================================
# QUESTION ANSWERING
# ==================================================

st.header("💬 Ask a Question")

question = st.text_input(
    "Enter your question about the document"
)


if st.button("Ask Question"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            # Search relevant document chunks
            results = search_documents(
                question,
                top_k=3
            )

            if not results:

                st.warning(
                    "No relevant documents found. "
                    "Please process a document first."
                )

            else:

                # Build context
                context = "\n\n".join(
                    result["text"]
                    for result in results
                )

                # Get source names
                sources = list(
                    dict.fromkeys(
                        result["source"]
                        for result in results
                    )
                )

                # Load QA model
                qa_model = load_qa_model()

                # Generate answer
                response = qa_model(
                    question=question,
                    context=context
                )

                answer = response["answer"]
                score = response["score"]

                st.subheader("✅ Answer")

                if score < 0.10:

                    st.info(
                        "The answer was not clearly found "
                        "in the uploaded document."
                    )

                else:

                    st.write(answer)

                st.subheader("📑 Sources")

                for source in sources:

                    st.write(f"• {source}")

        except Exception as error:

            st.error(
                f"Could not generate an answer: {error}"
            )
