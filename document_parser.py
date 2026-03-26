import os
import logging
from typing import Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

logger = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self, assets_dir: str = "data/assets", **kwargs):
        self.assets_dir = assets_dir
        os.makedirs(self.assets_dir, exist_ok=True)
        
        logger.info("Docling DocumentParser baslatiliyor...")
        
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def parse_document(self, file_path: str) -> Any:
        if not os.path.exists(file_path):
            logger.error("Dosya bulunamadi: %s", file_path)
            raise FileNotFoundError(f"{file_path} mevcut degil.")

        logger.info("%s isleniyor. Bu islem vakit alabilir...", file_path)
        
        try:
            conversion_result = self.converter.convert(file_path)
            document = conversion_result.document
            logger.info("Dokuman basariyla yapilandirildi ve gorseller cikarildi.")
            return document
        except Exception as e:
            logger.error("Dokuman islenirken hata olustu: %s", e)
            raise e