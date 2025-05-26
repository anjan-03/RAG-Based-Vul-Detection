import os
import tempfile
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import json
import fitz
import re
import subprocess
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DB_PATH = os.path.join(os.getcwd(), "dolos-py-db")
QUERY_PATH = "/home/codeai/codeql/python/ql/src/Security/CWE-078/CommandInjection.ql"
OUTPUT_FILE = os.path.join(os.getcwd(), "codeql_output.txt")
TEMP_UPLOAD = os.getcwd()  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = TEMP_UPLOAD

def create_codeql_db(sbom_dir):
    if os.path.exists(DB_PATH):
        print(f"Database '{DB_PATH}' already exists. Skipping creation.")
        return True

    print("Creating CodeQL database...")
    create_command = [
        "/opt/codeql/codeql", "database", "create", DB_PATH,
        "--language=python",
        "-s", sbom_dir,
        "--command=python3 app.py"
    ]

    try:
        result = subprocess.run(create_command, check=True, capture_output=True, text=True)
        print("Database creation successful!\n")
        return True
    except subprocess.CalledProcessError as e:
        print("Error during database creation:\n")
        print(e.stderr)
        return False

def run_codeql_query():
    print("Running CodeQL query ...")
    query_command = [
        "/opt/codeql/codeql", "query", "run",
        "--database=" + DB_PATH,
        QUERY_PATH
    ]

    try:
        with open(OUTPUT_FILE, "w") as outfile:
            subprocess.run(query_command, check=True, stdout=outfile, stderr=subprocess.STDOUT, text=True)
        print(f"Query output written to {OUTPUT_FILE}")
    except subprocess.CalledProcessError as e:
        print("Error during query execution. Check output.txt for details.")

def load_sbom(file_path):
    with open(file_path) as f:
        data = json.load(f)
    libraries = set()
    
    def find_libs(obj):
        if isinstance(obj, dict):
            if 'name' in obj and ('version' in obj or 'versionInfo' in obj):
                version = obj.get('versionInfo', obj.get('version', 'unknown'))
                name = obj['name']
                libraries.add((name, version))
            for v in obj.values():
                find_libs(v)
        elif isinstance(obj, list):
            for item in obj:
                find_libs(item)
    
    find_libs(data)
    return sorted(list(libraries))
    
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# def process_pdf(file_path):
#     with fitz.open(file_path) as doc:
#         text = "\n".join(page.get_text() for page in doc)
#     splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     return splitter.split_text(text)

def process_pdf(file_path):
    with fitz.open(file_path) as doc:
        text = "\n".join(page.get_text() for page in doc)

    def split_text(text, chunk_size=1000, chunk_overlap=200):        
        separators = ["\n\n", "\n", " ", ""]
        chunks = []
        current_splits = [text]
        for sep in separators:
            new_splits = []
            for s in current_splits:
                if sep:
                    new_splits.extend(s.split(sep))
                else:  # Final separator (character-level split)
                    new_splits.extend([s[i:i+chunk_size] for i in range(0, len(s), chunk_size)])
            current_splits = new_splits

        # Now merge with overlap
        merged_chunks = []
        current_chunk = []
        current_len = 0
        
        for split in current_splits:
            split = split.strip()
            if not split:
                continue
                
            if current_len + len(split) + 1 > chunk_size:
                if current_chunk:
                    merged_chunks.append(" ".join(current_chunk))
                    # Prepare overlap
                    overlap_start = max(0, len(current_chunk) - chunk_overlap)
                    current_chunk = current_chunk[overlap_start:]
                    current_len = sum(len(s) + 1 for s in current_chunk)
                    
            current_chunk.append(split)
            current_len += len(split) + 1  # +1 for space

        if current_chunk:
            merged_chunks.append(" ".join(current_chunk))
            
        # Final cleanup
        return [chunk for chunk in merged_chunks if chunk.strip()]

    return split_text(text, chunk_size=1000, chunk_overlap=200)

def extract_vuln_types(file_path):
    vuln_types = set()
    pattern = re.compile(r'([a-zA-Z0-9\s\-]+vulnerability)', re.IGNORECASE)
    with open(file_path, "r") as f:
        for line in f:
            matches = pattern.finditer(line)
            for match in matches:
                vuln_type = match.group(1).strip()
                vuln_types.add(vuln_type)
    return vuln_types

def generate_sbom(directory="Dolos_ML_CTF_Challenge", output_file=None):
    output_file = output_file or os.path.join(os.getcwd(), "sbom.json")
    try:
        command = ['syft', f'dir:{directory}', '-o', 'json']
        with open(output_file, 'w') as f:
            subprocess.run(command, stdout=f, check=True, text=True)
        print(f"SBOM successfully generated and saved to {output_file}")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error running syft: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Save uploaded PDF
    pdf_file = request.files['pdf']
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(pdf_file.filename))
    pdf_file.save(pdf_path)

    sbom_path = generate_sbom()
    if not sbom_path:
        return "SBOM generation failed.", 500

    result_lines = []
    sbom_dir = "Dolos_ML_CTF_Challenge/"

    if create_codeql_db(sbom_dir):
        run_codeql_query()

    # RAG Setup
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = OllamaLLM(model="mistral")
    sbom_libs = load_sbom(sbom_path)
    vuln_data = process_pdf(pdf_path)
    vuln_types = extract_vuln_types(OUTPUT_FILE)

    # FAISS VectorStores
    sbom_texts = [f"Library: {name}, Version: {version}" for name, version in sbom_libs]
    sbom_db = FAISS.from_texts(sbom_texts, embeddings)
    vuln_db = FAISS.from_texts(vuln_data, embeddings)
    sbom_retriever = sbom_db.as_retriever(search_kwargs={"k": 2})
    vuln_retriever = vuln_db.as_retriever(search_kwargs={"k": 3})

    def combined_retriever(query):
        return sbom_retriever.invoke(query) + vuln_retriever.invoke(query)

    template = """Answer based on context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    rag_chain = (
        {"context": combined_retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result_lines.append("<b>Checking vulnerabilities for all packages...</b><br>")
    vuln_results = []
    all_vuln_text = ""
    for name, version in sbom_libs:
        query = f"What vulnerabilities exist in library {name} version {version}?"
        try:
            answer = rag_chain.invoke(query)
            vuln_results.append((name, version, answer.strip()))
            all_vuln_text += answer.lower() + "\n"
        except Exception as e:
            vuln_results.append((name, version, f"Error: {e}"))

    result_lines.append("<hr><b>Vulnerabilities found:</b><br>")
    for name, version, vulns in vuln_results:
        result_lines.append(f"<b>Library:</b> {name} (version: {version})<br>")
        result_lines.append(f"<b>Vulnerabilities:</b> {vulns}<br><hr>")

    result_lines.append("<br><b>Vulnerability detected by CodeQL :</b>")
    for vt in vuln_types:
        result_lines.append(f"<br>- {vt}")

    matched = [vt for vt in vuln_types if vt.lower() in all_vuln_text]
    result_lines.append("<br><br><b>Matching vulnerabilities found:</b>")
    result_lines.append("<br>- " + "<br>- ".join(matched) if matched else "<br>Vulnerability detected by CodeQL may be a False Positive")

    return "<br>".join(result_lines)

app.run(host='0.0.0.0', port=5000, debug=False)