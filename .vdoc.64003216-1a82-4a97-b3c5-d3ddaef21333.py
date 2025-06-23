# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
%pip install openai  lancedb pyarrow pandas numpy matplotlib seaborn plotly pymupdf requests tqdm tenacity ipython dotenv langchain langchain-community langchain_openai  ipywidgets openpyxl  filetype
#
#
#
#
#
#| eval: false
%reset -f
#
#
#
#
#
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from langchain_openai import AzureChatOpenAI 
# Initialize LLM with higher temperature for creative question generation
llm_creative = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.7,
    max_tokens=500
)

llm_accurate = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.1,
    max_tokens=1000
)


#
#
#
#
#
#
#
#
import pandas as pd
import openpyxl
import json
def load_evaluations(file_path,json_path):
    # Load the Excel file
    df = pd.read_excel(file_path, sheet_name='extract from 2005 to Aug 2024', engine='openpyxl')
    
    # Filter evaluations from 2005 to August 2024
    filtered_df = df 
    
    # Create a nested structure
    evaluations = []
    for _, row in filtered_df.iterrows():
        evaluation = {
            "Title": row["Title"],
            "Year": str(row["Year"]),
            "Author": row["Author"],
            "Best Practices or Lessons Learnt": row["Best Practicesor Lessons Learnt"],
            "Date of Publication": str(row["Date of Publication"]),
            "Donor": row["Donor"],
            "Evaluation Brief": row["Evaluation Brief"],
            "Evaluation Commissioner": row["Evaluation Commissioner"],
            "Evaluation Coverage": row["Evaluation Coverage"],
            "Evaluation Period From Date": str(row["Evaluation Period From Date"]),
            "Evaluation Period To Date": str(row["Evaluation Period To Date"]),
            "Executive Summary": row["Executive Summary"],
            "External Version of the Report": row["External Version of the Report"],
            "Languages": row["Languages"],
            "Migration Thematic Areas": row["Migration Thematic Areas"],
            "Name of Project(s) Being Evaluated": row["Name of Project(s) Being Evaluated"],
            "Number of Pages Excluding annexes": row["Number of Pages Excluding annexes"],
            "Other Documents Included": row["Other Documents Included"],
            "Project Code": row["Project Code"],
            "Countries Covered": [country.strip() for country in str(row["Countries Covered"]).split(",")],
            "Regions Covered": row["Regions Covered"],
            "Relevant Crosscutting Themes": row["Relevant Crosscutting Themes"],
            "Report Published": row["Report Published"],
            "Terms of Reference": row["Terms of Reference"],
            "Type of Evaluation Scope": row["Type of Evaluation Scope"],
            "Type of Evaluation Timing": row["Type of Evaluation Timing"],
            "Type of Evaluator": row["Type of Evaluator"],
            "Level of Evaluation": row["Level of Evaluation"],
            "Documents": []
        }
        
        # Split the document-related fields by comma and create a list of dictionaries
        document_subtypes = str(row["Document Subtype"]).split(", ")
        file_urls = str(row["File URL"]).split(", ")
        file_descriptions = str(row["File description"]).split(", ")
        
        for subtype, url, description in zip(document_subtypes, file_urls, file_descriptions):
            document = {
                "Document Subtype": subtype,
                "File URL": url,
                "File description": description
            }
            evaluation["Documents"].append(document)
        
        evaluations.append(evaluation)

    ## dump data
    with open(json_path, 'w') as json_file:
        json.dump(evaluations, json_file, indent=4)

    
    return evaluations
#
#
#
library =load_evaluations("reference/Evaluation_repository.xlsx","reference/Evaluation_repository.json" )
#
#
#
#
#
#
import pandas as pd
import openpyxl
def load_iom_framework(excel_path: str) -> pd.DataFrame:
    """Load and validate IOM framework"""
    df = pd.read_excel(excel_path)
    
    # Validate expected columns
    required_columns = ['Objective', 'Outcome', 'Indicator']
    for col in required_columns:
        assert col in df.columns, f"Framework missing required column: {col}"
    
    return df
#
#
#
framework= load_iom_framework("reference/Strategic_Result_Framework.xlsx")    
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
from typing import List, Dict, Optional # Type hinting
import json
def load_evaluations(json_path: str) -> List[Dict]:
    """Load evaluation data from a JSON file
    
    Args:
        json_path: Path to the JSON file containing evaluation data
        
    Returns:
        List of evaluation dictionaries
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle both single evaluation and list of evaluations
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Invalid JSON structure - expected object or array")
            
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {json_path}: {e}")
        return []
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return []
#
#
#
#
#
# Load your   metadata
evaluation_data =  load_evaluations("reference/Evaluation_repository_test.json")
print(f"Attribute name is: {evaluation_data}")
print(type(evaluation_data))
#
#
#
#
#
import hashlib
def generate_id(text: str) -> str:
    """Generate a deterministic ID from text"""
    return hashlib.md5(text.encode()).hexdigest()
#
#
#
eval_id = generate_id( "aaa")
print({eval_id})
#
#
#
#
import time
import shutil
def force_delete_directory(path, max_retries=3, delay=1):
    """Robust directory deletion with retries and delay"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed to delete {path} after {max_retries} attempts: {e}")
                return False
            time.sleep(delay)
    return False

force_delete_directory(LANCE_DB_PATH)
#
#
#
#
#
from lancedb import connect
import numpy as np

def safe_get(d, key, default=None):
    """Safely get value from dict, handle NaN and missing keys"""
    value = d.get(key, default)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value

def initialise_knowledge_base(db, evaluation: Dict):
    """Store full documents without chunking (late chunking approach)"""

    import pyarrow as pa
    from typing import Optional

    # EvaluationModel Schema
    evaluation_schema = pa.schema([
        pa.field("evaluation_id", pa.string()),
        pa.field("title", pa.string()),
        pa.field("year", pa.string()),
        pa.field("author", pa.string()),
        pa.field("donor", pa.string()),
        pa.field("evaluation_commissioner", pa.string()),
        pa.field("migration_thematic_areas", pa.string()),
        pa.field("relevant_crosscutting_themes", pa.string()),
        pa.field("type_of_evaluation_timing", pa.string()),
        pa.field("type_of_evaluator", pa.string()),
        pa.field("level_of_evaluation", pa.string()),
        pa.field("scope", pa.list_(pa.string())),
        pa.field("geography", pa.list_(pa.string())),
        pa.field("summary", pa.string()),
        pa.field("evaluation_type", pa.string()),
        pa.field("population", pa.string()),
        pa.field("intervention", pa.string()),
        pa.field("comparator", pa.string()),
        pa.field("outcome", pa.string()),
        pa.field("methodology", pa.string()),
        pa.field("study_design", pa.string()),
        pa.field("sample_size", pa.string()),
        pa.field("data_collection_techniques", pa.string()),
        pa.field("evidence_strength", pa.string()),
        pa.field("limitations", pa.string())
    ])

    # DocumentModel Schema
    document_schema = pa.schema([
        pa.field("document_id", pa.string()),
        pa.field("url", pa.string()),
        pa.field("description", pa.string()),
        pa.field("evaluation_id", pa.string()),
        pa.field("content", pa.string()),
        pa.field("processed", pa.bool_()),
        pa.field("document_title", pa.string()),
        pa.field("document_type_infer", pa.string())
    ])

    # Create or open tables with explicit schema
    try:
        eval_table = db.create_table("evaluations", schema=evaluation_schema, exist_ok=True)
        doc_table = db.create_table("documents", schema=document_schema, exist_ok=True)
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise

    eval_id = generate_id(f"{evaluation['Title']}_{evaluation['Year']}")
    # Prepare the evaluation record
    eval_record = {
        "evaluation_id": eval_id,
        "title": evaluation["Title"],
        "year": evaluation["Year"],
        "author": safe_get(evaluation, "Author"),
        "donor": safe_get(evaluation, "Donor"),
        "evaluation_commissioner": safe_get(evaluation, "Evaluation Commissioner"),
        "migration_thematic_areas": safe_get(evaluation, "Migration Thematic Areas"),
        "relevant_crosscutting_themes": safe_get(evaluation, "Relevant Crosscutting Themes"),
        "type_of_evaluation_timing": safe_get(evaluation, "Type of Evaluation Timing"),
        "type_of_evaluator": safe_get(evaluation, "Type of Evaluator"),
        "level_of_evaluation": safe_get(evaluation, "Level of Evaluation"),
        "scope": safe_get(evaluation, "Type of Evaluation Scope"),
        "geography": safe_get(evaluation, "Countries Covered", []),

        ## rest will be filled later by the generate_evaluation_metadata()
        "summary": None,
        "evaluation_type": None,
        "population": None,
        "intervention": None,
        "comparator": None,
        "outcome": None,
        "methodology": None,
        "study_design": None,
        "sample_size": None,
        "data_collection_techniques": None,
        "evidence_strength": None,
        "limitations": None
    }
 
    # Insert evaluation (convert to RecordBatch first)
    eval_batch = pa.RecordBatch.from_pylist([eval_record], schema=evaluation_schema)  
    eval_table.add(eval_batch)  
    #print(f"Eval Record to insert: {eval_batch}")

    documents = []
    for doc in evaluation.get("Documents", []):
        doc_id = generate_id(doc["File URL"])
        documents.append({
            "document_id": doc_id,
            "url": doc["File URL"],            
            "description": doc["File description"],
            "evaluation_id": eval_id,
            "content": "",  # Will be filled when processed,
            "processed": False,            
            "document_title": None,
            "document_type_infer": None 
        })
    
   # print(f"Document to insert: {documents}")
    # Insert documents with schema validation
    if documents:
        try:
            #if "documents" in db.table_names():
            #    db.drop_table("documents")

            # doc_table = db.create_table("documents", schema=document_schema)
            
            schema_fields = [f.name for f in document_schema]
            documents_ordered = [{k: doc[k] for k in schema_fields} for doc in documents]
            # Create Arrow table ensuring schema match
            doc_data = pa.Table.from_pylist(documents_ordered)            
            # Align with expected schema
            aligned_data = doc_data.cast(document_schema)            
            # Add to LanceDB table
            doc_table.add(aligned_data)

        except Exception as e:
            print(f"Error adding documents: {e}")
            print(f"Document schema: \n  {doc_data.schema} \n \n")
            print(f"Expected schema: {document_schema}")
            raise

    return eval_id
#
#
#
LANCE_DB_PATH = "./lancedb"
db = connect(LANCE_DB_PATH)
for evaluation in evaluation_data:  # Assuming evaluation_data is a list
    initialise_knowledge_base(db, evaluation)
#
#
#
#
eval_table = db.open_table("evaluations")
#  Convert to Pandas DataFrame (recommended for display)
df = eval_table.to_pandas()
print(df)
#
#
#
#
LANCE_DB_PATH = "./lancedb"
from lancedb import connect
db = connect(LANCE_DB_PATH)
doc_table = db.open_table("documents")
#  Convert to Pandas DataFrame (recommended for display)
df = doc_table.to_pandas()
print(df)
# this table includes document_id, url, and evaluation_id
#
#
#
#
#
#
#
#
#
#
#
#
#
import os
import time
import random
import requests
from urllib.parse import urlparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import filetype  # For detecting file type
 
MAX_RETRIES = 5
RETRY_DELAY = 5  # base seconds
THREADS = 5
THROTTLE_DELAY_RANGE = (1, 3)  # Delay between downloads per thread

# Sample pool of common User-Agents
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 15_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
]


def download_documents(doc_table):
    pdf_root = Path(os.getenv("PDF_Library", "./PDF_Library"))

    def process_doc(doc):
        document_id = doc['document_id']
        url = doc['url']
        evaluation_id = doc['evaluation_id']

        try:
            file_name = os.path.basename(urlparse(url).path)
            print(f"processing {file_name}")
            if not file_name:
                file_name = f"{document_id}.pdf"  # fallback

            file_dir = pdf_root / str(evaluation_id)
            file_dir.mkdir(parents=True, exist_ok=True)
            file_path = file_dir / file_name

            if file_path.exists():
                return f"[✓] Skipped {file_name} (already exists)"

            # Rate throttling (add jitter)
            time.sleep(random.uniform(*THROTTLE_DELAY_RANGE))

            # Retry logic for downloading
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    headers = {'User-Agent': random.choice(USER_AGENTS)}
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code == 200:
                        with open(file_path, 'wb') as f:
                            f.write(response.content)
                        break
                    else:
                        raise Exception(f"Status {response.status_code}")
                except Exception as e:
                    if attempt == MAX_RETRIES:
                        return f"[✗] Failed to download {file_name} after {MAX_RETRIES} attempts: {e}"
                    time.sleep(RETRY_DELAY * attempt + random.uniform(0.5, 2.0))  # exponential backoff + jitter


            # Filetype validation
            kind = filetype.guess(file_path)
            actual_extension = file_path.suffix.lower().lstrip('.')

            # Check if it's already a PDF by detected type or file extension (case-insensitive)
            is_pdf = (kind and kind.extension.lower() == 'pdf') or actual_extension == 'pdf'

            if not is_pdf:
                pdf_path = file_path.with_suffix('.pdf')
                print(f"The file {file_path} is not a pdf and shall be converted")
                convert_file_to_pdf(file_path, pdf_path)
                file_path.unlink()  # remove original
                return f"[→] Converted {file_name} to PDF"
            return f"[✓] Downloaded {file_name}"

        except Exception as e:
            return f"[✗] Error processing {url}: {e}"


    # Fetch documents from table
    documents = doc_table.to_pandas().to_dict(orient="records")

    results = []
    with ThreadPoolExecutor(max_workers=THREADS) as executor:
        for result in tqdm(executor.map(process_doc, documents), total=len(documents)):
            results.append(result)

    for r in results:
        print(r)
#
#
#
#
#
#
#
#
#
#
#
#
#
import subprocess
import platform
from pathlib import Path
import shutil

def find_libreoffice_exec():
    """
    Finds the appropriate LibreOffice command based on OS.
    Returns path to LibreOffice CLI tool or raises an error.
    """
    system = platform.system()

    # Windows typically installs LibreOffice here
    if system == "Windows":
        possible_paths = [
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"
        ]
        for path in possible_paths:
            if Path(path).exists():
                return path
        raise FileNotFoundError("LibreOffice not found on Windows. Please install it or set it in PATH.")
    
    # On macOS, typically installed via brew or dmg
    elif system == "Darwin":
        possible_paths = [
            "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        ]
        for path in possible_paths:
            if Path(path).exists():
                return path
        # fallback to PATH
        return shutil.which("soffice") or shutil.which("libreoffice")

    # On Linux, assume it's installed via apt/yum/pacman
    elif system == "Linux":
        return shutil.which("libreoffice") or shutil.which("soffice")

    else:
        raise RuntimeError(f"Unsupported operating system: {system}")

def convert_file_to_pdf(input_path, output_path):
    """
    Converts Word, Excel, or PowerPoint file to PDF using LibreOffice in headless mode.
    Works on Windows, macOS, and Linux.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    try:
        libreoffice_exec = find_libreoffice_exec()
        if not libreoffice_exec or not Path(libreoffice_exec).exists():
            raise FileNotFoundError("LibreOffice executable not found.")

        # LibreOffice generates PDF in the same folder as input, same base name
        subprocess.run([
            libreoffice_exec,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_path.parent),
            str(input_path)
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        generated_pdf = input_path.with_suffix('.pdf')
        expected_pdf = output_path

        if generated_pdf.exists():
            generated_pdf.rename(expected_pdf)
        elif expected_pdf.exists():
            pass  # already saved there
        else:
            raise FileNotFoundError(f"PDF was not generated for: {input_path.name}")

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"LibreOffice failed: {e.stderr.decode().strip()}")
    except Exception as e:
        raise RuntimeError(f"Conversion error: {e}")
#
#
#
#
#
doc_table = db.open_table("documents")
os.environ["PDF_Library"] = "Evaluation_Library"
download_documents(doc_table)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# Initialize embeddings
import os
from dotenv import load_dotenv 
load_dotenv()
from langchain_openai import AzureOpenAIEmbeddings
embedding_model = AzureOpenAIEmbeddings(
    deployment=os.getenv("EMBEDDING_MODEL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION_EMBED"),
    chunk_size=1
)

test_embedding = embedding_model.embed_query("Hello world")
print(f"Embedding vector length: {len(test_embedding)}")

# LanceDB-compatible wrapper
class LangchainEmbeddingWrapper:
    def __init__(self, langchain_embedder):
        self._embedder = langchain_embedder

    def __call__(self, texts):
        return self._embedder.embed_documents(texts)
        
    def ndims(self):
        return self._dim

# Wrap and use
embedding_fn = LangchainEmbeddingWrapper(embedding_model)
#print("Embedding dimension:", embedding_fn.ndims())

vec = embedding_fn(["Hello world"])
print(f"Vector through lancedb dim: {len(vec[0])}")
print(embedding_fn(["Hello world"])[0])

#
#
#
print(dir(embedding_fn))
help(embedding_fn)
#
#
#
#
from pydantic import BaseModel
from lancedb.pydantic import Vector
import pyarrow as pa
pa_schema = pa.schema([
    pa.field("chunk_id", pa.string()),
    pa.field("document_id", pa.string()),
    pa.field("evaluation_id", pa.string()),
    pa.field("metadata", pa.string()),  # storing metadata dict as JSON string for simplicity
    pa.field("content", pa.string()),
    pa.field("vector", pa.list_(pa.float32(), embedding_dim)),  # vector as list of floats
])

from lancedb import connect
db = connect(LANCE_DB_PATH)
chunk_table = db.create_table("chunks", schema=pa_schema)

#
#
#
#
#
import os
import fitz  # PyMuPDF
import uuid
import time
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from lancedb.pydantic import Vector
import pandas as pd
from typing import Dict
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000  # characters
CHUNK_OVERLAP = 200  # characters
MAX_RETRIES = 5
RETRY_BACKOFF = 2  # seconds base (exponential)
os.environ["PDF_Library"] = "Evaluation_Library"

def process_documents_to_chunks(doc_table, chunk_table):
    PDF_LIBRARY = os.environ["PDF_Library"]

    def extract_text_from_pdf(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text() for page in doc)
            doc.close()
            return text.strip()
        except Exception as e:
            print(f"[✗] Failed to extract text from {pdf_path}: {e}")
            return None

    def sanitize_filename_from_url(url):
        file_name = Path(url.split("?")[0]).name
        return Path(file_name).stem + ".pdf"

    def chunk_text(text: str):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            keep_separator=True,
            is_separator_regex=False)
        return splitter.split_text(text)

    from lancedb.embeddings import get_registry
 
   # embedding_fn = get_registry().get("openai")()
    #embedding_dim = embedding_fn.ndims()
    embedding_dim = ""
    def embed_with_retry(text):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                # LanceDB embedding functions expect a list of texts
               # return embedding_fn.compute_query_embeddings([text])[0]
                return embedding_fn([text])[0]
            except Exception as e:
                wait_time = RETRY_BACKOFF ** attempt
                print(f"[!] Embedding failed (attempt {attempt}): {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(wait_time)

    def process_doc(doc: Dict) -> str:
        if doc.get("processed", False):
            return f"[✓] Already processed: {doc['document_id']}"

        url = doc["url"]
        print(f"Now processing: {url} \n")
        evaluation_id = doc["evaluation_id"]
        document_id = doc["document_id"]
        file_name = sanitize_filename_from_url(url)
        file_path = Path(PDF_LIBRARY) / evaluation_id / file_name

        if not file_path.exists():
            return f"[✗] Missing file: {file_path}"

        text = extract_text_from_pdf(file_path)
        if not text:
            return f"[✗] No text extracted: {file_path}"

        chunks = chunk_text(text)
        chunk_records = []

        for i, chunk in enumerate(chunks):
            try:
                chunk_id = f"{document_id}_{i}_{uuid.uuid4().hex[:6]}"
                vector = embed_with_retry(chunk)
                chunk_records.append({
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "evaluation_id": evaluation_id,
                    "metadata": json.dumps({"chunk_index": str(i)}),
                    "content": chunk,
                    "vector": vector
                })
            except Exception as e:
                print(f"[✗] Failed to embed chunk {i} of {document_id}: {e}")
                continue

        if chunk_records:
            chunk_table.add(pd.DataFrame(chunk_records))
            #doc_table.update({"document_id": document_id}, {"processed": True})
            doc_table.update(
                where=f"document_id = '{document_id}'",
                values={"processed": True}
            )
            return f"[→] Processed {document_id} with {len(chunk_records)} chunks"
        else:
            return f"[✗] No chunks processed for {document_id}"

    documents = doc_table.to_pandas().to_dict(orient="records")

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_doc, documents))

    ## Now building a text index for hybrid search
    import tantivy
    import lance
    chunk_table.create_fts_index("content")
    print("[✓] FTS index created on 'content'")

    for result in results:
        print(result)

#
#
#
#
#
#
LANCE_DB_PATH = "./lancedb"
from lancedb import connect
db = connect(LANCE_DB_PATH)
doc_table = db.open_table("documents")
chunk_table = db.open_table("chunks")
process_documents_to_chunks(doc_table, chunk_table)
#
#
#
#  Convert to Pandas DataFrame (recommended for display)
df = chunk_table.to_pandas()
print(df)
# this table includes document_id, url, and evaluation_id
#
#
#
#
#
#
#
#
#
#
#
#
#
#
import json
import time
import logging
from lancedb import connect
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from langchain_openai import AzureChatOpenAI 
llm_accurate = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.1,
    max_tokens=1000
)

# Set up logging
logging.basicConfig(filename="log/metadata_generation.log", level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Initialize DB
LANCE_DB_PATH = "./lancedb"
db = connect(LANCE_DB_PATH)
eval_table = db.open_table("evaluations")
chunk_table = db.open_table("chunks")


def call_llm_with_retries(prompt, max_retries=4, delay=2):
    for attempt in range(max_retries):
        try:
            response = llm_accurate.invoke([HumanMessage(content=prompt)])
            # Get the content directly from the AIMessage object
            raw_output = response.content.strip()

            # Try to parse the JSON
            try:
                parsed = json.loads(raw_output)
                # Validate the structure isn't just repeating the same value
                if isinstance(parsed, dict) and any(isinstance(v, list) and len(v) > 50 for v in parsed.values()):
                    raise JSONDecodeError("Output contains excessively repeated values", "", 0)
                return parsed
            except JSONDecodeError as e:
                # Try to fix common issues
                if raw_output.count('{') != raw_output.count('}'):
                    # Try to complete the JSON if it was cut off
                    if raw_output.count('{') > raw_output.count('}'):
                        raw_output += '}'
                    else:
                        raw_output = '{' + raw_output
                parsed = json.loads(raw_output)
                return parsed
                
        except JSONDecodeError as e:
            logging.warning(f"JSON parsing failed (attempt {attempt + 1}). Output:\n{raw_output}")
            if attempt == max_retries - 1:  # Last attempt
                # Try to salvage partial data
                try:
                    # Extract the valid part before the cutoff
                    valid_part = raw_output[:raw_output.rfind('}')+1]
                    return json.loads(valid_part)
                except:
                    raise RuntimeError(f"LLM returned malformed JSON that couldn't be repaired: {str(e)}")
        except Exception as e:
            logging.warning(f"LLM error (attempt {attempt + 1}): {str(e)}")
        time.sleep(delay * (2 ** attempt))
    raise RuntimeError("LLM call failed after retries.")


from lancedb.rerankers import RRFReranker
def get_context_for_eval(eval_row, query, chunk_table):
    
    reranker = RRFReranker()
    evaluation_id = eval_row["evaluation_id"]
    query_embedding = embedding_fn(query)
    results = chunk_table.search(query_embedding).where(
        f"evaluation_id = '{evaluation_id}'", prefilter=True
    ).limit(5).to_pandas()

    if results.empty:
        logging.warning(f"No chunks found for evaluation_id={evaluation_id}")
        return None

    context = "\n\n".join([
        f"Document:\n{row['content']}"
        for _, row in results.iterrows()
    ])
    return context
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#

from langchain.schema import HumanMessage

# Define query and embedding
query_descriptive = " Population, Intervention, Outcome, comparator"

def generate_metadata_for_evaluation_metadata_descriptive(eval_row, query_descriptive, chunk_table):
    """Process one evaluation row and return updated row with metadata."""
    evaluation_id = eval_row["evaluation_id"]

    context = get_context_for_eval(eval_row, query_descriptive, chunk_table)

    prompt = f"""

    You are an expert in public program and policy evaluation implementation. 
    
    Your task is to generate evaluation metadata related to a specific evaluation.

    Identify: 
    - Population:  The group of individuals or units (e.g., households, schools, firms) affected by the intervention. The target population shall be clearly defined (e.g., smallholder farmers, primary school students, unemployed youth) and it shall Include eligibility criteria (e.g., age, socioeconomic status, geographic location).

    - Intervention: The program, policy, or treatment whose effect is being evaluated. Describes the active component being tested (e.g., cash transfers, training workshops, new teaching methods). Should specify dosage, duration, and delivery mechanism.

    - Comparators: The counterfactual scenario, i.e. what would have happened without the intervention. Ideally involves a control group (if the study approach is randomized or quasi-experimental) that does not receive the intervention. Alternatively refers to "Business-as-usual" groups, placebo interventions, or different treatment arms.

    - Outcome:The measurable effects or endpoints used to assess the intervention’s impact. Includes primary outcomes (main indicators of interest, e.g., school enrollment rates, income levels) and secondary outcomes (e.g., health, empowerment). Should be specific, measurable, and time-bound (e.g., "child literacy scores after 12 months").


    Below are some existing Metadata on this evaluation:
    - Title: {eval_row.get('title')}
    - Year: {eval_row.get('year')}
    - Author: {eval_row.get('author')}
    - Evaluation Coverage: {eval_row.get('level_of_evaluation')}
    - Type of Evaluation Scope: {eval_row.get('scope')}
    - Type of Evaluation Timing: {eval_row.get('type_of_evaluation_timing')}
    - Thematic Areas: {eval_row.get('migration_thematic_areas')}
    - Cross cutting themes: {eval_row.get('relevant_crosscutting_themes')}
    - Countries: {', '.join(eval_row.get('geography', []))}

    and some Relevant Document Context:
    {context}

        Return ONLY valid JSON with the following structure:
    {{
        "summary":  "" ,
        "population": ["population1", "population2", ...],
        "intervention": ["intervention1", "intervention2", ...],
        "comparator": ["comparator1", "comparator2", ...],
        "outcome": ["outcome1", "outcome2", ...]
    }}

    Important Rules:
    1. Each array should contain no more than 10 items
    2. Items should be distinct and non-repetitive
    3. Return ONLY the JSON object, no additional text

    """

   # logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id}:\n{prompt}")
    
    logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id} ")
    try:
        metadata = call_llm_with_retries(prompt)
        eval_row.update(metadata)
        #logging.info(f"Processed evaluation_id={evaluation_id}")
        logging.info(f" {eval_row}")
        return eval_row
    except Exception as e:
        logging.error(f"LLM failed for evaluation_id={evaluation_id} | {str(e)}")
        return None

#
#
#

from langchain.schema import HumanMessage

# Define query and embedding
query_methodo = "Study design, Methodology, Sample, Data Collection"

def generate_metadata_for_evaluation_metadata_methodo(eval_row, query, chunk_table):
    """Process one evaluation row and return updated row with metadata."""
    evaluation_id = eval_row["evaluation_id"]

    context = get_context_for_eval(eval_row, query, chunk_table)

    prompt = f"""

    You are an expert in public program and policy evaluation implementation. 
    
    Your task is to generate evaluation metadata about the evaluation methodology: 
    - evaluation type
    - study_design
    - methodology (qualitative, quantitative, mixed methods)
    - sample_size: description of the sample used for the study
    - data_collection_techniques (e.g. surveys, interviews, focus groups, document review).
 
    Use the following list of evaluation type:
    - Formative Evaluation: Conducted during development/implementation to improve the programme, 
    - Process Evaluation: Examine implementation fidelity and operations, 
    - Outcome Evaluation: Measures short-term/intermediate results (between output and impact), 
    - Summative Evaluation: Assess overall effectiveness after completion, 
    - Impact Evaluation: Measure long-term effects and causal attribution, 
    
    Use the following list of study design types

    - "Observational - Cross-sectional: Data collected at a single point in time, no follow-up.",
    - "Observational - Cohort: Participants followed over time without experimental manipulation.",
    - "Experimental - Randomized Controlled Trial: Participants or units are randomly assigned to groups.",
    - "Experimental - Quasi-experimental: Includes comparison or time series design without randomization.",
    - "Hybrid Type 1: Primarily tests intervention effectiveness while collecting limited implementation data.",
    - "Hybrid Type 2: Simultaneously tests intervention  and implementation strategies.",
    - "Hybrid Type 3: Primarily tests implementation strategies while collecting limited intervention effectiveness data.",
    - "Case Study / Mixed-methods: In-depth exploration of implementation in one or few settings using qualitative and/or quantitative data." 

    Below are some existing Metadata on this evaluation:
    - Title: {eval_row.get('title')}
    - Year: {eval_row.get('year')}
    - Author: {eval_row.get('author')}
    - Evaluation Coverage: {eval_row.get('level_of_evaluation')}
    - Type of Evaluation Scope: {eval_row.get('scope')}
    - Type of Evaluation Timing: {eval_row.get('type_of_evaluation_timing')}
    - Thematic Areas: {eval_row.get('migration_thematic_areas')}
    - Cross cutting themes: {eval_row.get('relevant_crosscutting_themes')}
    - Countries: {', '.join(eval_row.get('geography', []))}
    - Summary: {eval_row.get('summary":  "" ,
    - Populations: {', '.join(eval_row.get('population', []))}
    - Interventions: {', '.join(eval_row.get('intervention', []))}
    - Compators: {', '.join(eval_row.get('comparator', []))}
    - Outcomes: {', '.join(eval_row.get('outcome', []))}


    and some Relevant Document Context:
    {context}

        Return ONLY valid JSON with the following structure:
    {{
        "methodology":  "" ,
        "evaluation_type":  ["evaluation_type1", "evaluation_type2", ...] ,
        "study_design":  "",
        "sample_size":  "" ,
        "population": ["population1", "population2", ...],
        "data_collection_techniques": ["data_collection_techniques1", "data_collection_techniques2", ...] 
    }}

    Important Rules:
    1. Each array should contain no more than 10 items
    2. Items should be distinct and non-repetitive
    3. Return ONLY the JSON object, no additional text

    """

   # logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id}:\n{prompt}")
    
    logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id} ")
    try:
        metadata = call_llm_with_retries(prompt)
        eval_row.update(metadata)
        #logging.info(f"Processed evaluation_id={evaluation_id}")
        logging.info(f" {eval_row}")
        return eval_row
    except Exception as e:
        logging.error(f"LLM failed for evaluation_id={evaluation_id} | {str(e)}")
        return None

#
#
#

from langchain.schema import HumanMessage

# Define query and embedding
query = "Evidence Limitation findings recommendations"

def generate_metadata_for_evaluation_metadata_evidence(eval_row, query, chunk_table):
    """Process one evaluation row and return updated row with metadata."""
    evaluation_id = eval_row["evaluation_id"]

    context = get_context_for_eval(eval_row, query, chunk_table)

    prompt = f"""

    You are an expert in public program and policy evaluation implementation. 
    
    Your task is to generate evaluation metadata about the evaluation output:
     - evidence_strength 
     - limitations 
 

    Below are some existing Metadata on this evaluation:
    - Title: {eval_row.get('title')}
    - Year: {eval_row.get('year')}
    - Author: {eval_row.get('author')}
    - Evaluation Coverage: {eval_row.get('level_of_evaluation')}
    - Type of Evaluation Scope: {eval_row.get('scope')}
    - Type of Evaluation Timing: {eval_row.get('type_of_evaluation_timing')}
    - Thematic Areas: {eval_row.get('migration_thematic_areas')}
    - Cross cutting themes: {eval_row.get('relevant_crosscutting_themes')}
    - Countries: {', '.join(eval_row.get('geography', []))}
    
    - Study Design: {eval_row.get('study_design":  ""
    - Cross cutting themes: {eval_row.get('evaluation_type":  ["evaluation_type1", "evaluation_type2", ...] ,

    and some Relevant Document Context:
    {context}

        Return ONLY valid JSON with the following structure:
    {{
        "evidence_strength":  "" ,
        "limitations":  ["limitations1", "limitations2", ...]  
    }}

    Important Rules:
    1. Each array should contain no more than 10 items
    2. Items should be distinct and non-repetitive
    3. Return ONLY the JSON object, no additional text

    """

   # logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id}:\n{prompt}")
    
    logging.info(f"Sending prompt to LLM for evaluation_id={evaluation_id} ")
    try:
        metadata = call_llm_with_retries(prompt)
        eval_row.update(metadata)
        #logging.info(f"Processed evaluation_id={evaluation_id}")
        logging.info(f" {eval_row}")
        return eval_row
    except Exception as e:
        logging.error(f"LLM failed for evaluation_id={evaluation_id} | {str(e)}")
        return None

#
#
#
def generate_evaluation_metadata(eval_table, chunk_table, batch_size=50):
    """Main function to generate metadata in batches and update the table."""
    all_rows = eval_table.to_pandas().to_dict(orient="records")
    enriched_data = []
    total = len(all_rows)
    
    logging.info(f"processing {total} evaluations!")
    success = 0
    skipped = 0

    for i in range(0, len(all_rows), batch_size):
        batch = all_rows[i:i+batch_size]
        enriched_batch = []
             
        for row in batch:
            enriched = generate_metadata_for_evaluation_metadata_descriptive(row, query, chunk_table)
            if enriched:
                enriched_batch.append(enriched)
                row=enriched_batch
                success += 1
            else:
                skipped += 1

            enriched = generate_metadata_for_evaluation_metadata_methodo(row, query, chunk_table)
            if enriched:
                enriched_batch.append(enriched)
                row=enriched_batch
                success += 1
            else:
                skipped += 1

            enriched = generate_metadata_for_evaluation_metadata_evidence(row, query, chunk_table)
            if enriched:
                enriched_batch.append(enriched)
                success += 1
            else:
                skipped += 1        

        if enriched_batch:
           # eval_table.add(enriched_batch)
            enriched_data.extend(enriched_batch)
            logging.info(f"Added batch of {len(enriched_batch)} rows.")

        time.sleep(2)  # prevent overloading LLM/API

    print(f"[✓] Metadata generation complete. Success: {success}, Skipped: {skipped}, Total: {total}")
    logging.info(f"Final counts — Success: {success}, Skipped: {skipped}, Total: {total}")

    # Save to file
   # with open("evaluations_metadata.json", "w", encoding="utf-8") as f:
    #    json.dump(enriched_data, f, ensure_ascii=False, indent=2)
    print("[✓] Metadata generation complete. Saved to app/evaluations_metadata.json")

    return enriched_data

#
#
#
#
enriched_data= generate_evaluation_metadata(eval_table, chunk_table)
#
#
#
#
#
#
#
#
#
#
# Define the list of experts on impact - outcome - organisation
q_experts = [
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the Strategic Impact: ---Attaining favorable protection environments---: i.e., finding or recommendations that require a change in existing policy and regulations. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the Strategic Impact: ---Realizing rights in safe environments---: i.e., finding or recommendations that require a change in existing policy and regulations. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the Strategic Impact: ---Empowering communities and achieving gender equality--- : i.e., finding or recommendations that require a change in existing policy and regulations. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the Strategic Impact: ---Securing durable solutions--- : i.e., finding or recommendations that require a change in existing policy and regulations. [/INST]",

   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: ---Access to territory registration and documentation ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Status determination ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Protection policy and law---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Gender-based violence ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Child protection ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Safety and access to justice ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Community engagement and women's empowerment ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Well-being and basic needs ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Sustainable housing and settlements ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Healthy lives---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Education ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Clean water sanitation and hygiene ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Self-reliance, Economic inclusion, and livelihoods ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Voluntary repatriation and sustainable reintegration ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Resettlement and complementary pathways---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on the specific Operational Outcome: --- Local integration and other local solutions ---, i.e. finding or recommendations that require a change that needs to be implemented in the field as an adaptation or change of current activities. [/INST]", 


   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on Organizational Enablers related to Systems and processes, i.e. elements that require potential changes in either management practices, technical approach, business processes, staffing allocation or capacity building. [/INST]",
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on Organizational Enablers related to Operational support and supply chain, i.e. elements that require potential changes in either management practices, technical approach, business processes, staffing allocation or capacity building. [/INST]" ,
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on Organizational Enablers related to People and culture, i.e. elements that require potential changes in either management practices, technical approach, business processes, staffing allocation or capacity building. [/INST]" ,
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on Organizational Enablers related to External engagement and resource mobilization, i.e. elements that require potential changes in either management practices, technical approach, business processes, staffing allocation or capacity building. [/INST]" ,
   "<s> [INST] Instructions: Act as a public program evaluation expert working for UNHCR. Your specific area of expertise and focus is strictly on Organizational Enablers related to Leadership and governance, i.e. elements that require potential changes in either management practices, technical approach, business processes, staffing allocation or capacity building. [/INST]" 
]

# Predefined knowledge extraction questions
q_questions = [
    " List, as bullet points, all findings and evidences in relation to your specific area of expertise and focus. ",
    " Explain, in relation to your specific area of expertise and focus, what are the root causes for the situation. " ,
    " Explain, in relation to your specific area of expertise and focus, what are the main risks and difficulties here described. ",
    " Explain, in relation to your specific area of expertise and focus, what what can be learnt. ",
    " List, as bullet points, all recommendations made in relation to your specific area of expertise and focus. "#,
    # "Indicate if mentionnend what resource will be required to implement the recommendations made in relation to your specific area of expertise and focus. ",
    # "List, as bullet points, all recommendations made in relation to your specific area of expertise and focus that relates to topics  or activities recommended to be discontinued. ",
    # "List, as bullet points, all recommendations made in relation to your specific area of expertise and focus that relates to topics or activities recommended to be scaled up. " 
    # Add more questions here...
]

## Additional instructions!
q_instr = """
</s>
[INST]  
Keep your answer grounded in the facts of the contexts. 
If the contexts do not contain the facts to answer the QUESTION, return {NONE} 
Be concise in the response and  when relevant include precise citations from the contexts. 
[/INST] 
"""
#
#
#
#
#
qa_questions = [
    "What was the intervention type?",
    "What outcomes were observed?",
    "What population was targeted?",
    "What geographic area was covered?",
    "How strong is the evidence?",
]

def generate_qas(text):
    prompt = f"""Extract answers to the following questions from the evaluation:
    {json.dumps(qa_questions)}
    
    Text: {text[:3000]}
    """
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return completion.choices[0].message.content

df_docs["qa"] = df_docs["text"].apply(generate_qas)
#
#
#
#
#
# Sample hybrid search query
query = "What works best to improve health outcomes for displaced persons?"

query_embedding = get_text_embedding(query)
results = table.search(query_embedding).limit(5).to_list()

for result in results:
    print(result['metadata'])
    print(result['text'][:500])
#
#
#
#
def query_evidence(question: str, table: lancedb.db.LanceTable) -> Dict:
    """Enhanced query with hybrid search and evidence grading"""
    try:
        # Hybrid search
        results = hybrid_search(table, question, limit=7)
        
        if results.empty:
            return {"answer": "No relevant evidence found.", "sources": []}
        
        context = "\n\n".join([
            f"Document {i+1} (Relevance: {row.get('combined_score', 0):.2f}):\n{row['text']}\n"
            for i, row in results.iterrows()
        ])
        
        # Evidence-based answer generation
        prompt = f"""
        You are an evidence specialist answering questions about IOM programs.
        Use ONLY the provided context from evaluation reports.
        For each claim in your answer, cite the document number it came from.
        
        Question: {question}
        
        Context:
        {context}
        
        Provide:
        1. A direct answer to the question
        2. Strength of evidence (High/Medium/Low)
        3. Any limitations or caveats
        4. List of sources with relevance scores
        """
        
        response = openai.ChatCompletion.create(
            engine=config.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        
        answer = response.choices[0].message.content
        sources = [
            {"url": url, "score": score}
            for url, score in zip(results['url'], results.get('combined_score', 0))
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "search_scores": results[['_distance', '_score', 'combined_score']].to_dict()
        }
    
    except Exception as e:
        print(f"Query error: {str(e)}")
        return {"error": str(e)}

#
#
#
#
def extract_structured_info(table, iom_framework):
    """Extract structured information from reports using the IOM Results Framework"""
    # Generate questions based on the IOM framework
    questions = generate_questions_from_framework(iom_framework)
    
    # Store extracted information
    extracted_data = []
    
    # Process each question
    for question in questions:
        print(f"Processing question: {question}")
        
        # Search for relevant chunks
        results = table.search(generate_embeddings([question])[0]).limit(10).to_pandas()
        
        # Combine relevant chunks as context
        context = "\n\n".join(results["text"].tolist())
        
        # Use Azure OpenAI to extract structured answer
        prompt = f"""
        Based on the following evaluation report excerpts, answer the question with structured information.
        Provide your answer in JSON format with the following structure:
        {{
            "question": "the question being asked",
            "answer": "the concise answer",
            "intervention_type": "type of intervention mentioned",
            "population": "target population mentioned",
            "outcome": "outcome measured",
            "geography": "geographic location if mentioned",
            "evidence_strength": "strength of evidence (high/medium/low)"
        }}

        Question: {question}
        Context: {context}
        """
        
        response = openai.ChatCompletion.create(
            engine=config["azure_openai_chat_deployment"],
            messages=[{"role": "user", "content": prompt}],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"]
        )
        
        try:
            answer = json.loads(response.choices[0].message.content)
            answer["source_urls"] = results["url"].unique().tolist()
            extracted_data.append(answer)
        except json.JSONDecodeError:
            print(f"Failed to parse answer for question: {question}")
    
    return pd.DataFrame(extracted_data)

def generate_questions_from_framework(framework_df):
    """Generate questions based on the IOM Results Framework"""
    questions = []
    
    # Example questions based on common evaluation themes
    for _, row in framework_df.iterrows():
        questions.extend([
            f"What interventions has IOM implemented to achieve {row['Objective']}?",
            f"What evidence exists for the effectiveness of interventions targeting {row['Outcome']}?",
            f"What populations have been targeted by interventions aiming for {row['Indicator']}?",
            f"What geographic areas have seen interventions related to {row['Objective']}?",
            f"What methodologies have been used to evaluate interventions for {row['Outcome']}?"
        ])
    
    # Add some general evaluation questions
    questions.extend([
        "What are the most effective interventions for migrant livelihood improvement?",
        "What evidence exists for cash-based interventions in migration contexts?",
        "What are common challenges in implementing migration programs?",
        "What evaluation methodologies are most commonly used in IOM evaluations?",
        "What gaps exist in the evidence base for migration interventions?"
    ])
    
    return list(set(questions))  # Remove duplicates

#
#
#
def hybrid_search(table: lancedb.db.LanceTable, query: str, limit: int = 10) -> pd.DataFrame:
    """Perform hybrid (vector + full-text) search"""
    # Generate query embedding
    query_embedding = generate_embeddings_batch([query])[0]
    
    # Perform hybrid search
    results = table.search(query_embedding, query_string=query)\
                 .limit(limit)\
                 .to_pandas()
    
    # Score normalization (simple example)
    if not results.empty:
        max_vector_score = results["_distance"].max()
        max_fts_score = results["_score"].max()
        
        if max_vector_score > 0 and max_fts_score > 0:
            results["combined_score"] = (
                0.7 * (results["_distance"] / max_vector_score) +
                0.3 * (results["_score"] / max_fts_score)
            )
            results = results.sort_values("combined_score", ascending=False)
    
    return results

def extract_structured_info(table: lancedb.db.LanceTable, iom_framework: pd.DataFrame) -> pd.DataFrame:
    """Enhanced information extraction with hybrid search"""
    questions = generate_questions_from_framework(iom_framework)
    extracted_data = []
    
    for question in questions:
        try:
            # Hybrid search for relevant chunks
            results = hybrid_search(table, question, limit=15)
            
            if results.empty:
                continue
                
            context = "\n\n".join(results["text"].tolist())
            sources = results["url"].unique().tolist()
            
            # Structured extraction prompt
            prompt = f"""
            Extract structured information from this evaluation report context to answer the question.
            Return ONLY valid JSON with this structure:
            {{
                "question": "the question",
                "answer": "concise answer",
                "intervention_type": ["type1", "type2"],
                "population": ["group1", "group2"],
                "outcome": ["outcome1", "outcome2"],
                "geography": ["location1", "location2"],
                "evidence_strength": "high/medium/low",
                "source_urls": ["url1", "url2"]
            }}
            
            Question: {question}
            Context: {context}
            """
            
            response = openai.ChatCompletion.create(
                engine=config.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                response_format={ "type": "json_object" }
            )
            
            answer = json.loads(response.choices[0].message.content)
            answer["source_urls"] = sources
            extracted_data.append(answer)
            
        except Exception as e:
            print(f"Error processing question '{question}': {str(e)}")
            continue
    
    return pd.DataFrame(extracted_data)

#
#
#
#
#
def extract_information(text):
    # Use Azure OpenAI to extract information
    response = openai.Completion.create(
        engine="your-completion-engine",
        prompt=f"Extract information from: {text}",
        max_tokens=150
    )
    return response.choices[0].text.strip()

df['structured_info'] = df['text'].apply(extract_information)
#
#
#
#
#
#
#
#
#
#
#
#
def generate_insights(df):
    # Add your insight generation logic here
    return df

df = generate_insights(df)
#
#
#
#
#
#
#
def identify_patterns(df):
    # Add your pattern identification logic here
    return df

df = identify_patterns(df)
#
#
#
#
#
#
#
def generate_deliverables(df):
    # Generate Q&A dataset
    qa_dataset = df[['question', 'answer']]

    # Generate synthesis report
    synthesis_report = df.describe()

    # Generate visual evidence map
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='outcome', y='population', size='sample_size')
    plt.title('Visual Evidence Map')
    plt.show()

    return qa_dataset, synthesis_report

qa_dataset, synthesis_report = generate_deliverables(df)
#
#
#
#
#
# Convert QA to structured fields (intervention, outcome, population, etc.)
qa_df = pd.json_normalize(df_docs["qa"].apply(json.loads))

# Bubble Map Example
fig = px.scatter(qa_df, x="geography", y="outcome",
                 size="sample_size", color="intervention",
                 hover_name="file_name",
                 title="Evidence Bubble Map")
fig.show()

# Heatmap Example
heatmap_df = pd.crosstab(qa_df["intervention"], qa_df["outcome"])
sns.heatmap(heatmap_df, annot=True, cmap="coolwarm")
#
#
#
def create_interactive_visualizations(extracted_data: pd.DataFrame):
    """Enhanced visualization functions"""
    # Prepare data
    df = extracted_data.explode("source_urls")
    
    # Evidence Strength Distribution
    strength_dist = df['evidence_strength'].value_counts().reset_index()
    fig1 = px.bar(
        strength_dist,
        x='evidence_strength',
        y='count',
        title='Distribution of Evidence Strength'
    )
    
    # Interventions by Geography
    fig2 = px.treemap(
        df,
        path=['geography', 'intervention_type'],
        title='Interventions by Geographic Region'
    )
    
    # Evidence Timeline (if dates available)
    if 'date' in df.columns:
        fig3 = px.line(
            df.groupby('date').size().reset_index(name='count'),
            x='date',
            y='count',
            title='Evidence Publication Timeline'
        )
    else:
        fig3 = None
    
    return fig1, fig2, fig3

#
#
#
#
#
def visualize_evidence_map(extracted_data):
    """Create interactive visualizations of the evidence map"""
    
    # Prepare data for visualization
    df = extracted_data.explode("source_urls")
    
    # Bubble map: Interventions by outcome and evidence strength
    fig1 = px.scatter(
        df, 
        x="outcome", 
        y="intervention_type", 
        size="evidence_strength",  # This would need to be mapped to numeric values
        color="population",
        hover_name="answer",
        title="Evidence Map: Interventions by Outcome and Population"
    )
    fig1.update_layout(height=800)
    
    # Heatmap: Evidence concentration by intervention and outcome
    heatmap_data = df.groupby(['intervention_type', 'outcome']).size().unstack().fillna(0)
    fig2 = px.imshow(
        heatmap_data,
        labels=dict(x="Outcome", y="Intervention Type", color="Number of Studies"),
        title="Evidence Concentration Heatmap"
    )
    
    # Gap map: Missing evidence
    all_interventions = df['intervention_type'].unique()
    all_outcomes = df['outcome'].unique()
    complete_grid = pd.MultiIndex.from_product([all_interventions, all_outcomes], names=['intervention_type', 'outcome'])
    gap_data = df.groupby(['intervention_type', 'outcome']).size().reindex(complete_grid, fill_value=0).reset_index()
    gap_data['has_evidence'] = gap_data[0] > 0
    
    fig3 = px.scatter(
        gap_data,
        x="outcome",
        y="intervention_type",
        color="has_evidence",
        title="Evidence Gap Map (Red = Missing Evidence)"
    )
    
    return fig1, fig2, fig3

def generate_synthesis_report(extracted_data):
    """Generate a narrative synthesis report of findings"""
    prompt = f"""
    You are an evaluation specialist analyzing evidence from IOM evaluation reports.
    Below is structured data extracted from multiple evaluation reports:
    
    {extracted_data.to_json()}
    
    Write a comprehensive synthesis report that:
    1. Summarizes key findings across interventions
    2. Identifies areas with strong evidence
    3. Highlights evidence gaps
    4. Provides recommendations for future evaluations
    5. Suggests high-priority research areas
    
    Structure your report with clear sections and bullet points for readability.
    """
    
    response = openai.ChatCompletion.create(
        engine=config["azure_openai_chat_deployment"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # Slightly more creative for synthesis
        max_tokens=3000
    )
    
    return response.choices[0].message.content

#
#
#
#
#
#
#
#
qa_dataset.to_csv('qa_dataset.csv', index=False)
synthesis_report.to_csv('synthesis_report.csv')
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
