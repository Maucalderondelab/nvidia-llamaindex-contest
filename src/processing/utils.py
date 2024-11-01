from typing import List, Dict, Any, Tuple
from .parser import DocumentContent, ImageContent
import textwrap

class ContentValidator:
    """Validates content before embedding"""
    
    @staticmethod
    def validate_text(text: str) -> bool:
        """Validate text content"""
        if not text or len(text.strip()) < 10:  # Arbitrary minimum length
            return False
        return True
    
    @staticmethod
    def validate_image(image: ImageContent) -> bool:
        """Validate image content"""
        # Add specific validation for Neva-22b requirements
        return True

class EmbeddingPreprocessor:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.validator = ContentValidator()

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunks = textwrap.wrap(text, self.chunk_size)
        
        # Add overlap
        if len(chunks) > 1:
            overlapped_chunks = []
            for i in range(len(chunks)):
                if i == 0:
                    overlapped_chunks.append(chunks[i])
                else:
                    # Add overlap from previous chunk
                    overlapped_chunks.append(
                        chunks[i-1][-self.chunk_overlap:] + chunks[i]
                    )
            return overlapped_chunks
        return chunks

    def prepare_for_text_embedding(self, processed_docs: List[DocumentContent]) -> List[Dict[str, Any]]:
        """Prepares text content for NV-embedding-ada"""
        text_chunks = []
        
        for doc in processed_docs:
            if doc.error:
                continue
                
            for page in doc.pages:
                if not self.validator.validate_text(page.text):
                    continue
                    
                chunks = self.chunk_text(page.text)
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_data = {
                        "text": chunk,
                        "page_number": page.page_number,
                        "chunk_index": chunk_idx,
                        "document_path": str(doc.file_path),
                        "metadata": doc.metadata
                    }
                    text_chunks.append(chunk_data)
                    
        return text_chunks

    def prepare_for_image_processing(self, processed_docs: List[DocumentContent]) -> List[Dict[str, Any]]:
        """Prepares images for Neva-22b processing"""
        image_data = []
        
        for doc in processed_docs:
            if doc.error:
                continue
                
            for page in doc.pages:
                for image in page.images:
                    if not self.validator.validate_image(image):
                        continue
                        
                    img_data = {
                        "image_path": image.path,
                        "page_number": page.page_number,
                        "image_index": image.index,
                        "document_path": str(doc.file_path),
                        "metadata": doc.metadata,
                        "image_size": image.size,
                        "format": image.format
                    }
                    image_data.append(img_data)
                    
        return image_data