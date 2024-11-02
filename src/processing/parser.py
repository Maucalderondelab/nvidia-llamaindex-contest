from dataclasses import dataclass
from pathlib import Path
# typing imports
from typing import Dict, List, Optional, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
import logging
import fitz
import camelot
import ghostscript
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
        supported_formats: List[str] = ['.pdf'],
        max_workers: int = 4,
        min_image_size: Tuple[int, int] = (100, 100),
        supported_image_formats: List[str] = ['jpeg', 'png', 'jpg'],
        table_accuracy_threshold: int = 80
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.supported_formats = supported_formats
        self.max_workers = max_workers
        self.min_image_size = min_image_size
        self.supported_image_formats = supported_image_formats
        self.table_accuracy_threshold = table_accuracy_threshold
        self._setup_directories()

    def _setup_directories(self) -> None:
        """Create the necesary directories"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tables").mkdir(parents=True, exist_ok=True)

    def get_documents_paths(self) -> List[Path]:
        """Get all supported documents from input directory"""
        all_files = []
        for format in self.supported_formats:
            format = format.lstrip('.')
            all_files.extend(self.input_dir.glob(f"**/*.{format}"))
        return all_files
    def _process_image(
        self, 
        image_bytes: bytes, 
        page_num: int, 
        img_idx: int,
        base_image: Dict,
        doc_num: int
    ) -> Optional[ImageContent]:
        """Process and validate a single image"""
        try:
            img_hash = hashlib.md5(image_bytes).hexdigest()
            image = Image.open(io.BytesIO(image_bytes))
            
            if (image.size[0] < self.min_image_size[0] or 
                image.size[1] < self.min_image_size[1] or
                base_image["colorspace"] not in [1, 3]):
                return None
                
            image_format = base_image.get("ext", "png")
            if image_format not in self.supported_image_formats:
                image_format = "png"
                
            image_path = self.output_dir / "images" / f"doc_{doc_num}_page_{page_num}_img_{img_idx}.{image_format}"
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
        
    def _extract_tables(self, pdf_path: str, page_num: int, doc_num: int) -> List[Dict[str, Union[str, int]]]:
        """Extract tables from a PDF page using camelot"""
        try:
            tables = camelot.read_pdf(
                pdf_path,
                pages=str(page_num + 1),
                flavor='stream',
                layout_kwargs={'detect_vertical': False},
                suppress_stdout=True
            )
            
            processed_tables = []
            
            if len(tables) > 0:
                for idx, table in enumerate(tables):
                    if table.parsing_report['accuracy'] > self.table_accuracy_threshold:
                        try:
                            table_path = self.output_dir / "tables" / f"doc_{doc_num}_page_{page_num}_table_{idx}.csv"
                            table.to_csv(str(table_path))
                            processed_tables.append({
                                "path": str(table_path),
                                "page": page_num,
                                "index": idx,
                                "accuracy": table.parsing_report['accuracy']
                            })
                        except Exception as e:
                            logger.warning(f"Failed to save table {idx} from page {page_num}: {e}")
                            continue
            
            return processed_tables
            
        except Exception as e:
            if "Fatal" in str(e):
                logger.warning(f"Ghostscript error in page {page_num}, trying alternative method...")
                try:
                    tables = camelot.read_pdf(
                        pdf_path,
                        pages=str(page_num + 1),
                        flavor='lattice',
                        suppress_stdout=True
                    )
                    processed_tables = []
                    if len(tables) > 0:
                        for idx, table in enumerate(tables):
                            if table.parsing_report['accuracy'] > self.table_accuracy_threshold:
                                try:
                                    table_path = self.output_dir / "tables" / f"doc_{doc_num}_page_{page_num}_table_{idx}.csv"
                                    table.to_csv(str(table_path))
                                    processed_tables.append({
                                        "path": str(table_path),
                                        "page": page_num,
                                        "index": idx,
                                        "accuracy": table.parsing_report['accuracy']
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to save table {idx} from page {page_num}: {e}")
                                    continue
                    return processed_tables
                except:
                    logger.warning(f"Both table extraction methods failed for page {page_num}, skipping...")
                    return []
            else:
                logger.warning(f"Failed to extract tables from {pdf_path} and page {page_num}: {e}")
                return []
            
    def process_page(self, page: fitz.Page, doc: fitz.Document, page_num: int, doc_num: int) -> PageContent:
        """Process a single page of a PDF document"""
        try:
            text = page.get_text()
            images = []
            seen_hashes = set()
            
            for img_idx, img in enumerate(page.get_images()):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    if processed_img := self._process_image(
                        base_image["image"],
                        page_num,
                        img_idx,
                        base_image,
                        doc_num
                    ):
                        if processed_img.hash not in seen_hashes:
                            images.append(processed_img)
                            seen_hashes.add(processed_img.hash)
                            
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
            
            tables = self._extract_tables(doc.name, page_num, doc_num)
            
            return PageContent(
                text=text,
                tables=tables,
                images=images,
                page_number=page_num
            )
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            raise

    def process_document(self, file_path: Path, doc_num: int) -> DocumentContent:
        """Process a single document"""
        try:
            doc = fitz.open(str(file_path))
            pages = []
            
            for page_num in range(len(doc)):
                page_content = self.process_page(doc[page_num], doc, page_num, doc_num)
                pages.append(page_content)
            
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "page_count": len(doc),
                "document_number": doc_num
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
                metadata={"document_number": doc_num},
                error=str(e)
            )

    def process_all_documents(self) -> List[DocumentContent]:
        """Process all documents in the input directory"""
        document_paths = self.get_documents_paths()
        processed_documents = []
        
        document_paths = sorted(document_paths)
        
        for doc_num, doc_path in enumerate(document_paths, start=1):
            try:
                logger.info(f"Processing document {doc_num}: {doc_path.name}")
                result = self.process_document(doc_path, doc_num)
                processed_documents.append(result)
            except Exception as e:
                logger.error(f"Failed to process document: {e}")
        
        return processed_documents

    def __del__(self):
        """Cleanup when parser is destroyed"""
        try:
            ghostscript.cleanup()
        except:
            pass