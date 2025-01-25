import faiss
import json
import os
import pandas as pd

from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from PyPDF2 import PdfReader
from docx import Document as DocxReader
from striprtf.striprtf import rtf_to_text
from pytesseract import image_to_string
from PIL import Image


# Initialize the vector store
def initialize_vector_store():
    # Path to save/load the FAISS index
    index_path = "faiss_index"
    
    # Initialize an empty FAISS index if it doesn't exist
    if os.path.exists(index_path):
        # Load existing FAISS index
        index = faiss.read_index(f"{index_path}/index")
        with open(f"{index_path}/docstore.json", "r") as f:
            docstore = InMemoryDocstore.load(f)
        with open(f"{index_path}/index_to_docstore_id.json", "r") as f:
            index_to_docstore_id = json.load(f)
    else:
        # Create a new FAISS index
        index = faiss.IndexFlatL2(1536)  # Assuming embedding dimensions = 1536
        docstore = InMemoryDocstore({})
        index_to_docstore_id = {}

    # Initialize the FAISS vector store
    vector_store = FAISS(
        embedding_function=OpenAIEmbeddings(),
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    
    return vector_store


# Add documents to the vector store
def add_document_to_vector_store(file_path, vector_store):
    from langchain.document_loaders import UnstructuredFileLoader
    
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    
    # Add documents to vector store
    vector_store.add_documents(documents)
    
    # Save the updated FAISS index and docstore
    faiss.write_index(vector_store.index, "faiss_index/index")
    vector_store.docstore.save("faiss_index/docstore.json")
    with open("faiss_index/index_to_docstore_id.json", "w") as f:
        json.dump(vector_store.index_to_docstore_id, f)


def extract_content_with_unstructured(file_path):
    """
    Extracts content from any file using Unstructured.
    """
    try:
        # Use UnstructuredFileLoader for general file extraction
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
        content = "\n".join([doc.page_content for doc in documents])
        return content
    except Exception as e:
        return f"Error processing file: {str(e)}"


def extract_content_from_file(file_path):
    """Extracts text content from various file formats."""
    extension = os.path.splitext(file_path)[-1].lower()

    try:
        if extension == ".txt":
            # Handle plain text files
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

        elif extension == ".pdf":
            # Handle PDF files
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() for page in reader.pages)

        elif extension == ".docx":
            # Handle Word documents
            doc = DocxReader(file_path)
            return "\n".join([para.text for para in doc.paragraphs])

        elif extension == ".rtf":
            # Handle RTF files
            with open(file_path, "r", encoding="utf-8") as f:
                return rtf_to_text(f.read())

        elif extension in [".jpg", ".jpeg", ".png"]:
            # Handle image files (OCR)
            image = Image.open(file_path)
            return image_to_string(image)

        elif extension in [".xlsx", ".xls"]:
            # Handle Excel files
            df = pd.read_excel(file_path)
            return df.to_string()

        else:
            return f"Unsupported file extension: {extension}"

    except Exception as e:
        return f"Error processing {file_path}: {e}"
