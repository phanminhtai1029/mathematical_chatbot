"""
Process PDF and Save Vector Store
"""

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

from config import *


def process_pdf(pdf_path, vectorstore_path):
    """Process PDF and save vector store"""
    print("Starting PDF processing...")
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    
    # Load PDF
    print(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")
    
    # Split into chunks
    print("Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    print(f"Saving vector store to: {vectorstore_path}")
    vectorstore.save_local(vectorstore_path)
    
    print("PDF processing complete!")
    print(f"Vector store saved to: {vectorstore_path}")

def main():
    """Main function"""
    print("=" * 50)
    print("PDF PROCESSOR")
    print("=" * 50)
    
    # Set your PDF path here
    pdf_path = DEFAULT_PDF
    vectorstore_path = DEFAULT_VECTORS
    
    try:
        process_pdf(pdf_path, vectorstore_path)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()