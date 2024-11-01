from pathlib import Path
from src.processing.parser import DocumentParser
from src.processing.utils import EmbeddingPreprocessor
import logging
from rich.console import Console
from rich.progress import track
import time

# Set up rich console for better output
console = Console()

def main():
    console.print("[bold blue]Starting Document Processing Pipeline[/bold blue]")
    
    # Define paths
# Define your document directories
    DOCUMENT_PATH = Path("/home/mauricio/Documents/Projects/nvidia-llamaindex-contest/AI Neighborhood Experience Explorer/data/database")  # Where your PDFs are stored
    OUTPUT_PATH = Path("/home/mauricio/Documents/Projects/nvidia-llamaindex-contest/AI Neighborhood Experience Explorer/data/processed")      # Where processed files will be saved    
    console.print(f"\n📁 Document Path: {DOCUMENT_PATH}")
    console.print(f"📁 Output Path: {OUTPUT_PATH}\n")
    
    # Ensure directories exist
    DOCUMENT_PATH.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    console.print("[bold green]Initializing processors...[/bold green]")
    parser = DocumentParser(
        input_dir=DOCUMENT_PATH,
        output_dir=OUTPUT_PATH,
        min_image_size=(224, 224)
    )
    preprocessor = EmbeddingPreprocessor(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Process documents
    console.print("\n[bold yellow]Starting document processing...[/bold yellow]")
    start_time = time.time()
    
    processed_docs = parser.process_all_documents()
    
    # Print processing results
    console.print(f"\n✅ Processed {len(processed_docs)} documents in {time.time() - start_time:.2f} seconds")
    
    # Print document details
    for doc in processed_docs:
        console.print(f"\n[bold cyan]Document: {doc.file_path.name}[/bold cyan]")
        if doc.error:
            console.print(f"[bold red]Error: {doc.error}[/bold red]")
            continue
            
        total_pages = len(doc.pages)
        total_images = sum(len(page.images) for page in doc.pages)
        total_tables = sum(len(page.tables) for page in doc.pages)
        
        console.print(f"├── Pages: {total_pages}")
        console.print(f"├── Images detected: {total_images}")
        console.print(f"└── Tables detected: {total_tables}")
    
    # Prepare content for models
    console.print("\n[bold yellow]Preparing content for models...[/bold yellow]")
    
    text_chunks = preprocessor.prepare_for_text_embedding(processed_docs)
    console.print(f"✅ Created {len(text_chunks)} text chunks for embedding")
    
    # Print sample of text chunks
    if text_chunks:
        console.print("\n[bold cyan]Sample text chunk:[/bold cyan]")
        console.print(f"├── Page: {text_chunks[0]['page_number']}")
        console.print(f"├── Chunk index: {text_chunks[0]['chunk_index']}")
        console.print(f"└── Text preview: {text_chunks[0]['text'][:100]}...")
    
    image_data = preprocessor.prepare_for_image_processing(processed_docs)
    console.print(f"\n✅ Prepared {len(image_data)} images for processing")
    
    # Print sample of image data
    if image_data:
        console.print("\n[bold cyan]Sample image data:[/bold cyan]")
        console.print(f"├── Page: {image_data[0]['page_number']}")
        console.print(f"├── Size: {image_data[0]['image_size']}")
        console.print(f"└── Format: {image_data[0]['format']}")
    
    console.print("\n[bold green]Processing pipeline completed![/bold green]")
    
    return text_chunks, image_data

if __name__ == "__main__":
    try:
        text_chunks, image_data = main()
    except Exception as e:
        console.print(f"\n[bold red]Error: {str(e)}[/bold red]")
        raise