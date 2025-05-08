"""
Generate FAISS vector embeddings for an insurance policy PDF document in S3.
This script downloads a PDF from S3, processes it, creates embeddings using Bedrock,
and uploads the resulting vector files back to S3.
"""

import boto3
import os
import logging
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
PDF_KEY = "docs/sample_policy_doc_AU1234.pdf"  # S3 path to the PDF file
POLICY_ID = "AU1234"  # Policy ID to be used in vector paths

# Create temporary directory for processing
TMP_DIR = "/tmp/vector_processing"
os.makedirs(TMP_DIR, exist_ok=True)
LOCAL_PDF_PATH = f"{TMP_DIR}/{POLICY_ID}.pdf"
VECTOR_DIR = f"{TMP_DIR}/{POLICY_ID}"
os.makedirs(VECTOR_DIR, exist_ok=True)

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime', region_name=REGION)
s3 = boto3.client('s3', region_name=REGION)

logger.info(f"Downloading PDF {PDF_KEY} from S3 bucket {BUCKET_NAME}")
try:
    # Download PDF from S3
    s3.download_file(BUCKET_NAME, PDF_KEY, LOCAL_PDF_PATH)
    logger.info(f"Successfully downloaded PDF to {LOCAL_PDF_PATH}")
except Exception as e:
    logger.error(f"Failed to download PDF: {e}")
    raise

try:
    # Load and process PDF
    logger.info(f"Loading PDF document from {LOCAL_PDF_PATH}")
    loader = PyPDFLoader(LOCAL_PDF_PATH)
    documents = loader.load()
    logger.info(f"PDF loaded successfully. Document count: {len(documents)}")
    
    # Split text into chunks
    logger.info("Splitting text into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Text split into {len(chunks)} chunks")
    
    # Create embeddings and FAISS index
    logger.info("Initializing Bedrock embeddings")
    embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock)
    
    logger.info("Creating FAISS index from documents")
    faiss_index = FAISS.from_documents(chunks, embeddings)
    
    # Save FAISS index locally
    logger.info(f"Saving FAISS index to {TMP_DIR}")
    faiss_index.save_local(TMP_DIR, index_name="policydoc")
    
    # Rename files to match expected pattern
    os.rename(f"{TMP_DIR}/policydoc.faiss", f"{VECTOR_DIR}/policydoc_faiss.faiss")
    os.rename(f"{TMP_DIR}/policydoc.pkl", f"{VECTOR_DIR}/policydoc_pkl.pkl")
    
    # Upload to S3
    logger.info("Uploading vector files to S3")
    s3.upload_file(
        f"{VECTOR_DIR}/policydoc_faiss.faiss", 
        BUCKET_NAME, 
        f"vectors/policydoc/{POLICY_ID}/policydoc_faiss.faiss"
    )
    s3.upload_file(
        f"{VECTOR_DIR}/policydoc_pkl.pkl", 
        BUCKET_NAME, 
        f"vectors/policydoc/{POLICY_ID}/policydoc_pkl.pkl"
    )
    
    logger.info(f"Vector embeddings for policy {POLICY_ID} created and uploaded successfully")
    
    # Verify files were uploaded correctly
    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=f"vectors/policydoc/{POLICY_ID}/"
    )
    
    if 'Contents' in response:
        logger.info("Uploaded files:")
        for item in response['Contents']:
            logger.info(f"- {item['Key']} ({item['Size']} bytes)")
    else:
        logger.warning("No files found in the destination path after upload")
        
except Exception as e:
    logger.error(f"Error generating embeddings: {e}")
    raise

logger.info("Script execution completed")
