from dataclasses import dataclass
from pathlib import Path
# typing imports
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
import fitz
import camelot
from PIL import Image
import hashlib
import io


# We setup the logging
logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

@dataclass
class ImageContent:
    """"Dataclass for image content"""
    path: str
    page_number: int
    index: int
    hash: str
    size: Tuple[int, int]
    format: str

@dataclass
class PageContent:
    """Data class to store content extracted from a single page"""
    text: str
    tables: List[Dict[str, Union[str, int]]]
    images: List[ImageContent]
    page_number: int

@dataclass
class DocumentContent:
    """Data class to store processed document content"""
    file_path: Path
    pages: List[PageContent]
    metadata: Dict[str, str]
    error: Optional[str] = None

class DocumentParser:
    """Parser for handling PDF documents with text, tables and images"""
    def __init__(
        self, 
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        supported_formats: List[str] = ['pdf'],
        max_workers: int = 4,
        min_image_size: Tuple[int, int] = (100, 100),  # Minimum size to process
        supported_image_formats: List[str] = ['jpeg', 'png', 'jpg']
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.supported_formats = supported_formats
        self.max_workers = max_workers
        self.min_image_size = min_image_size
        self.supported_image_formats = supported_image_formats
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create the necesary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)

    def get_documents_paths(self) -> List[Path]:
        """"Get all supported documents from input directory"""
        all_files = []
        for format in self.supported_formats:
            all_files.extend(self.input_dir.glob(f"**/*.{format}"))
        return all_files
    def _process_image(
        self, 
        image_bytes: bytes, 
        page_num: int, 
        img_idx: int,
        base_image: Dict
    ) -> Optional[ImageContent]:
        """Process and validate a single image"""
        try:
            # Generate hash to avoid duplicates
            img_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Load image for validation
            image = Image.open(io.BytesIO(image_bytes))
            
            # Basic validation
            if (image.size[0] < self.min_image_size[0] or 
                image.size[1] < self.min_image_size[1] or
                base_image["colorspace"] not in [1, 3]):  # 1=gray, 3=rgb
                return None
                
            # Save image
            image_format = base_image.get("ext", "png")
            if image_format not in self.supported_image_formats:
                image_format = "png"
                
            image_path = self.output_dir / "images" / f"page_{page_num}_img_{img_idx}.{image_format}"
            image.save(image_path, format=image_format)
            
            return ImageContent(
                path=str(image_path),
                page_number=page_num,
                index=img_idx,
                hash=img_hash,
                size=image.size,
                format=image_format
            )
            
        except Exception as e:
            logger.warning(f"Failed to process image {img_idx} on page {page_num}: {e}")
            return None
    def process_page(slef, page:fitz.Page, doc: fitz.Document, page_num: int) -> PageContent:
        """
        Process a single page of a PDF document
        
        Args:
            page: PDF page object
            doc: PDF document object
            page_num: Page number
            
        Returns:
            PageContent object containing extracted content
        """

        try:
            text = page.get_text()
            tables = self.extract_tables(doc, page_num)

            return PageContent(
                text = text,
                tables = tables,
                page_number = page_num
            )
        
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return []
        
    def _extract_tables(self, pdf_path: str, page_num: int) -> List[Dict[str, Union[str, int]]]:
        """Extract tables from a PDF file using camelot"""
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num+1))
            processed_tables = []

            for idx, table in enumerate(tables):
                table_path = self.output_dir / "tables" / f"page_{page_num}_table_{idx}.csv"
                table.to_csv(table_path)
                processed_tables.append({
                    "path": str(table_path),
                    "page": page_num,
                    "table": idx
                })

            return processed_tables
        except Exception as e:
            logger.warning(f"Error extracting tables from {pdf_path} and page {page_num}: {e}")
            return []

    def process_document(self, file_path: Path) -> DocumentContent:
        """
        Process a single document
        
        Args:
            file_path: Path to the document
            
        Returns:
            DocumentContent object containing all extracted content
        """
        try:
            doc = fitz.open(file_path)
            pages = []

            # Process each page of the document
            for page_num, page in enumerate(doc):
                page_content = self.process_page(page, doc, page_num)
                pages.append(page_content)

                # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "page_count": len(doc)
            }
            
            doc.close()
            return DocumentContent(
                file_path=file_path,
                pages=pages,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            return DocumentContent(
                file_path=file_path,
                pages=[],
                metadata={},
                error=str(e)
            )
            
    def process_all_documents(self) -> List[DocumentContent]:
        """
        Process all documents in the input directory
        
        Returns:
            List of DocumentContent objects
        """
        document_paths = self.get_documents_paths()
        processed_documents = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_doc = {
                executor.submit(self.process_document, doc_path): doc_path 
                for doc_path in document_paths
            }
            
            for future in future_to_doc:
                try:
                    result = future.result()
                    processed_documents.append(result)
                except Exception as e:
                    logger.error(f"Failed to process document: {e}")
                    
        return processed_documents