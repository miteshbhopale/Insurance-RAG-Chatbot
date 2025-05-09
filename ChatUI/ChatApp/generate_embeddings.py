"""
Automated embedding generator for insurance policy PDF documents in S3.

This script:
1. Scans the S3 docs folder for PDF files
2. Extracts policy IDs from filenames (based on a naming pattern)
3. Processes each PDF to create embeddings using Bedrock
4. Uploads the resulting vector files back to S3, organized by policy ID
"""

import boto3
import os
import logging
import re
import argparse
from pathlib import Path
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration variables
REGION = "us-east-1"
BUCKET_NAME = "rag-bot-source-834215301031-us-east-1"
PDF_PREFIX = "docs/"  # S3 path prefix for PDF files
VECTOR_PREFIX = "vectors/policydoc/"  # S3 path prefix for vector storage
TMP_DIR = "/tmp/vector_processing"  # Temporary processing directory

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings for insurance policy PDFs")
    parser.add_argument("--policy-id", help="Process only a specific policy ID")
    parser.add_argument("--pdf-pattern", default=r".*_([A-Z]{2}\d+)\.pdf$", 
                      help="Regex pattern to extract policy ID from filename")
    parser.add_argument("--chunk-size", type=int, default=1000, 
                      help="Size of text chunks for processing")
    parser.add_argument("--chunk-overlap", type=int, default=200, 
                      help="Overlap between text chunks")
    return parser.parse_args()

def extract_policy_id(filename, pattern=r".*_([A-Z]{2}\d+)\.pdf$"):
    """Extract policy ID from filename using regex pattern."""
    match = re.match(pattern, filename)
    if match:
        return match.group(1)
    return None

def list_policy_pdfs(s3_client, bucket, prefix, policy_id=None, pattern=r".*_([A-Z]{2}\d+)\.pdf$"):
    """List all PDF files in the S3 prefix that match the policy ID pattern."""
    logger.info(f"Scanning for PDFs in s3://{bucket}/{prefix}")
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    pdf_files = []
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            if key.lower().endswith('.pdf'):
                extracted_id = extract_policy_id(key, pattern)
                if extracted_id:
                    if policy_id is None or extracted_id == policy_id:
                        pdf_files.append((key, extracted_id))
                        logger.info(f"Found policy PDF: {key} (Policy ID: {extracted_id})")
                else:
                    logger.warning(f"Could not extract policy ID from {key}")
    
    return pdf_files

def process_pdf(s3_client, bedrock_client, bucket, pdf_key, policy_id, chunk_size, chunk_overlap):
    """Process a single PDF file to generate and store embeddings."""
    logger.info(f"Processing PDF {pdf_key} for policy ID {policy_id}")
    
    # Create temporary directories
    os.makedirs(TMP_DIR, exist_ok=True)
    local_pdf_path = f"{TMP_DIR}/{policy_id}.pdf"
    vector_dir = f"{TMP_DIR}/{policy_id}"
    os.makedirs(vector_dir, exist_ok=True)
    
    try:
        # Download PDF from S3
        logger.info(f"Downloading PDF from S3 bucket {bucket}")
        s3_client.download_file(bucket, pdf_key, local_pdf_path)
        logger.info(f"Successfully downloaded PDF to {local_pdf_path}")
        
        # Load and process PDF
        logger.info(f"Loading PDF document from {local_pdf_path}")
        loader = PyPDFLoader(local_pdf_path)
        documents = loader.load()
        logger.info(f"PDF loaded successfully. Document count: {len(documents)}")
        
        # Split text into chunks
        logger.info("Splitting text into chunks")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Text split into {len(chunks)} chunks")
        
        # Create embeddings and FAISS index
        logger.info("Initializing Bedrock embeddings")
        embeddings = BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1", 
            client=bedrock_client
        )
        
        logger.info("Creating FAISS index from documents")
        faiss_index = FAISS.from_documents(chunks, embeddings)
        
        # Save FAISS index locally
        logger.info(f"Saving FAISS index to {vector_dir}")
        faiss_index.save_local(vector_dir, index_name="policydoc")
        
        # Upload to S3
        logger.info("Uploading vector files to S3")
        s3_client.upload_file(
            f"{vector_dir}/policydoc.faiss", 
            bucket,
            f"{VECTOR_PREFIX}{policy_id}/policydoc_faiss.faiss"
        )
        s3_client.upload_file(
            f"{vector_dir}/policydoc.pkl", 
            bucket,
            f"{VECTOR_PREFIX}{policy_id}/policydoc_pkl.pkl"
        )
        
        logger.info(f"Vector embeddings for policy {policy_id} created and uploaded successfully")
        
        # Verify files were uploaded correctly
        response = s3_client.list_objects_v2(
            Bucket=bucket,
            Prefix=f"{VECTOR_PREFIX}{policy_id}/"
        )
        
        if 'Contents' in response:
            logger.info("Uploaded files:")
            for item in response['Contents']:
                logger.info(f"- {item['Key']} ({item['Size']} bytes)")
        else:
            logger.warning("No files found in the destination path after upload")
            
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_key}: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(local_pdf_path):
            os.remove(local_pdf_path)
        for filename in os.listdir(vector_dir):
            os.remove(os.path.join(vector_dir, filename))

def main():
    """Main execution function."""
    args = parse_args()
    
    # Initialize AWS clients
    s3 = boto3.client('s3', region_name=REGION)
    bedrock = boto3.client('bedrock-runtime', region_name=REGION)
    
    # Create base temporary directory
    os.makedirs(TMP_DIR, exist_ok=True)
    
    # List PDFs in S3 that match the policy pattern
    policy_pdfs = list_policy_pdfs(
        s3, 
        BUCKET_NAME, 
        PDF_PREFIX,
        policy_id=args.policy_id,
        pattern=args.pdf_pattern
    )
    
    if not policy_pdfs:
        logger.warning("No matching policy PDFs found")
        return
    
    logger.info(f"Found {len(policy_pdfs)} policy PDFs to process")
    
    # Process each PDF file
    successful = 0
    failed = 0
    
    for pdf_key, policy_id in policy_pdfs:
        logger.info(f"Processing {pdf_key} (Policy ID: {policy_id})")
        
        result = process_pdf(
            s3, 
            bedrock, 
            BUCKET_NAME, 
            pdf_key, 
            policy_id,
            args.chunk_size,
            args.chunk_overlap
        )
        
        if result:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"Processing complete. Successfully processed {successful} PDFs. Failed: {failed}")

if __name__ == "__main__":
    main()
