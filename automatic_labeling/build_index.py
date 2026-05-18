#!/usr/bin/python3
import ast
import chromadb
import nbformat
import re
import shutil

from pathlib import Path
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

db_path = Path("../chroma_db")
if db_path.exists():
    shutil.rmtree(db_path)
client = chromadb.PersistentClient(path="../chroma_db")
collection = client.get_or_create_collection("assignments")


BASE = Path("path/to/all/assignment/files") # TODO


def process_notebook(file_path:Path):
    """
    Split Markdown by cells. Simplified version of "process_assignments.py". (not included in repository)
    
    :param file_path: Path to Jupyter notebook file
    :type entries: Path
    """
    instances = []
    
    nb = nbformat.read(file_path, as_version=4)
    
    for cell in nb.cells:
        content = cell.source.strip()
        
        if not content:
            continue
        instances.append({
            "type": cell.cell_type,
            "content": content
        })
        
    return instances


def process_markdown(file_path:Path):
    """
    Split Markdown by headings (#, ##, etc.). Simplified version of "process_assignments.py". (not included in repository)
    
    :param file_path: Path to Markdown file
    :type entries: Path
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    instances = []
    
    heading_pattern = r'^(#{1,6})\s+(.*)'
    
    current_heading = ""
    current_content = []
    
    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            if current_heading or current_content:
                instance = current_heading.strip() + "\n" if current_heading else ""
                instance += "\n".join(current_content).strip()
                instances.append(instance)
                
            current_heading = match.group(0).strip()
            current_content = []
        else:
            current_content.append(line)
    
    instance = current_heading.strip() + "\n" if current_heading else ""
    instance += "\n".join(current_content).strip()
    instances.append(instance)
    
    return instances
    

def _extract_code(node, lines):
    return "\n".join(lines[node.lineno - 1: node.end_lineno]).strip()


def process_python(file_path:Path):
    """
    Split Python file by AST logic.
    
    :param file_path: Path to Python file
    :type entries: Path
    
    :param semester: Assignment the file belongs to (a1-a7)
    :type entries: str
    """
    source = Path(file_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines()
    
    instances = []
    buffer = []
    
    def flush_buffer():
        if buffer:
            instances.append("\n".join(buffer).strip())
            buffer.clear()
    
    for node in tree.body:
        # Top-level functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            flush_buffer()
            instances.append(_extract_code(node, lines))
            
        # Classes --> get methods individually
        elif isinstance(node, ast.ClassDef):
            flush_buffer()
            for item in node.body:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        instances.append(_extract_code(item, lines))
    
        # Imports
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            flush_buffer()
            instances.append(_extract_code(node, lines))
            
        else:
            # Free code
            buffer.append(_extract_code(node, lines))
            
    flush_buffer()

    if not instances:
        instances.append(source.strip())
    
    return instances


def chunk(text:str, size:int = 2000):
    '''
    Splits string into chunks of character-length < size while preserving words.
    
    :param text: Text to be chunked
    :param size: Max. character length of chunks
    '''
    tokens = re.findall(r'\S+|\s+', text)
    chunks = []
    current = ""
    
    for token in tokens:
        if len(current) + len(token) > size:
            chunks.append(current)
            current = token
        else:
            current += token
            
    if current:
        chunks.append(current)
        
    return chunks

for folder_path in BASE.iterdir():
    texts = None
    chunks = []
    
    for file_path in folder_path.iterdir():
        if file_path.suffix == ".md":
            instances = process_markdown(file_path)
            for instance in instances:
                chunks.extend(chunk(instance))
        elif file_path.suffix == ".ipynb":
            instances = process_notebook(file_path)
            for instance in instances:
                if instance['type'] == 'markdown':
                    chunks.extend(chunk(instance['content']))
                else:
                    chunks.append(instance['content'])
        elif file_path.suffix == ".py":
            chunks.extend(process_python(file_path))
            
    embeddings = model.encode(chunks)
    
    for i, (c, e) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents = [c],
            embeddings = [e.tolist()],
            ids = [f"{folder_path.name}_{i}"],
            metadatas = [{"assignment": folder_path.name}]
        )