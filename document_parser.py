import os
import logging
from typing import Any
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)

from src.utils import load_config

logger = logging.getLogger(__name__)


class DocumentParser:
    def __init__(self, assets_dir: str = "data/assets", **kwargs):
        self.assets_dir = assets_dir
        os.makedirs(self.assets_dir, exist_ok=True)

        logger.info("Docling DocumentParser baslatiliyor...")

        cfg = load_config()
        parser_cfg = cfg.get("document_parser", {})

        accelerator_device = parser_cfg.get("accelerator_device", "cpu")
        accelerator_threads = parser_cfg.get("accelerator_threads", 4)
        generate_picture_images = parser_cfg.get(
            "generate_picture_images",
            True,
        )

        device_map = {
            "auto": AcceleratorDevice.AUTO,
            "cpu": AcceleratorDevice.CPU,
            "cuda": AcceleratorDevice.CUDA,
            "mps": AcceleratorDevice.MPS,
            "xpu": AcceleratorDevice.XPU,
        }

        accelerator = device_map.get(
            str(accelerator_device).lower(),
            AcceleratorDevice.CPU,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = generate_picture_images
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=accelerator_threads,
            device=accelerator,
        )

        logger.info(
            "Docling accelerator ayari: device=%s, threads=%s, "
            "generate_picture_images=%s",
            accelerator.value,
            accelerator_threads,
            generate_picture_images,
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
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
            logger.info(
                "Dokuman basariyla yapilandirildi ve gorseller cikarildi."
            )
            return document
        except Exception as e:
            logger.error("Dokuman islenirken hata olustu: %s", e)
            raise e