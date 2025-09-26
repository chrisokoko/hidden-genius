"""
PDF file processor with progressive fallback: pdfplumber -> OCR -> LLM cleanup.
"""

import logging
import time
import re
import os
from pathlib import Path
import pdfplumber

from ..core.base import BaseProcessor
from ..core.types import ProcessorResult


# Setup logging
logger = logging.getLogger(__name__)


class PDFProcessor(BaseProcessor):
    """Extract text from PDF files with progressive fallback strategies."""

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        return file_path.suffix.lower() == '.pdf'

    def process(self, file_path: Path) -> ProcessorResult:
        """Process PDF: extract text with pdfplumber, then always send to LLM for formatting."""
        start_time = time.time()

        try:
            # Extract raw text with pdfplumber
            logger.info(f"Extracting text from {file_path.name}")
            raw_text = self._extract_with_pdfplumber(file_path)

            # Check if text is garbled
            is_garbled = self._is_text_garbled(raw_text)

            # Always send to LLM for final formatting
            if is_garbled:
                logger.info(f"Text appears garbled, using vision-enhanced LLM processing for {file_path.name}")
                extraction_method = "llm_vision"
            else:
                logger.info(f"Text is clean, using text-only LLM processing for {file_path.name}")
                extraction_method = "llm_text"

            # Process through LLM (handles page-by-page processing internally)
            text = self._extract_with_llm(raw_text, file_path, use_vision=is_garbled)

            if not text or len(text.strip()) < 10:
                logger.warning(f"Very little text extracted from {file_path.name}")
                text = "[WARNING: Very little text extracted from PDF]"

            duration = time.time() - start_time
            logger.info(f"✅ Completed: {file_path.name} ({duration:.1f}s, {len(text)} chars, method: {extraction_method})")

            return ProcessorResult(
                success=True,
                text=text,
                source_file=file_path,
                processor_type=f"pdf_{extraction_method}",
                processing_time=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Failed to process PDF {file_path.name}: {e}"
            logger.error(error_msg)

            return ProcessorResult(
                success=False,
                text="",
                source_file=file_path,
                processor_type="pdf",
                processing_time=duration,
                error_message=str(e)
            )

    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """Extract text using pdfplumber."""
        text_pages = []

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()

                    if page_text and page_text.strip():
                        text_pages.append(f"--- Page {page_num} ---")
                        text_pages.append(page_text.strip())
                        text_pages.append("")
                    else:
                        logger.warning(f"No text found on page {page_num} of {file_path.name}")
                        text_pages.append(f"--- Page {page_num} ---")
                        text_pages.append("[NO TEXT FOUND ON THIS PAGE]")
                        text_pages.append("")

                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num} in {file_path.name}: {e}")
                    text_pages.append(f"--- Page {page_num} (ERROR) ---")
                    text_pages.append(f"[ERROR: Could not extract text from this page: {e}]")
                    text_pages.append("")

        return "\n".join(text_pages).strip()


    def _convert_page_to_image(self, file_path: Path, page_number: int):
        """Convert a specific PDF page to a high-quality image."""
        try:
            from pdf2image import convert_from_path

            # Convert only the specific page (page_number is 0-indexed)
            images = convert_from_path(
                file_path,
                dpi=300,  # High DPI for clear text
                first_page=page_number + 1,  # pdf2image uses 1-indexed pages
                last_page=page_number + 1
            )

            if images:
                return images[0]  # Return the PIL Image
            else:
                raise Exception(f"No image generated for page {page_number + 1}")

        except ImportError:
            raise Exception("pdf2image not installed. Install with: pip install pdf2image")
        except Exception as e:
            raise Exception(f"Image conversion failed for page {page_number + 1}: {e}")

    def _encode_image_for_api(self, image):
        """Convert PIL image to base64 for Claude API."""
        try:
            import base64
            from io import BytesIO

            # Convert PIL image to bytes
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            # Encode as base64
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            return base64_image

        except Exception as e:
            raise Exception(f"Image encoding failed: {e}")

    def _extract_with_vision(self, page_text: str, file_path: Path, page_number: int) -> str:
        """Process a single page using vision + text analysis."""
        import anthropic

        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise Exception("CLAUDE_API_KEY not found in environment variables")

        client = anthropic.Anthropic(api_key=api_key)

        # Convert page to image for visual analysis (convert to 0-indexed)
        page_image = self._convert_page_to_image(file_path, page_number - 1)
        base64_image = self._encode_image_for_api(page_image)

        # Create multimodal message with both text and image
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You have both extracted text from a PDF page and an image of that same page.

TASK: Create markdown that exactly matches the visual layout shown in the image.

Use the extracted text for content, but use the IMAGE to determine:
- Which text should be headers (# ## ###) based on visual size/prominence
- What's bold, italic, or emphasized based on visual styling
- Table structures and column alignment as shown visually
- Bullet points, indentation, and list formatting
- Spatial relationships and visual hierarchy

Output only the markdown that matches the visual layout.

EXTRACTED TEXT:"""
                    },
                    {
                        "type": "text",
                        "text": page_text
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_image
                        }
                    }
                ]
            }
        ]

        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=8192,
            temperature=0,
            messages=messages
        )

        return response.content[0].text.strip()

    def _extract_text_only(self, page_text: str) -> str:
        """Process a single page using text-only analysis."""
        import anthropic

        api_key = os.getenv('CLAUDE_API_KEY')
        if not api_key:
            raise Exception("CLAUDE_API_KEY not found in environment variables")

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Extract the exact text from this PDF page and convert it to markdown format.

CRITICAL REQUIREMENTS:
- Use the EXACT words, phrases, and text that appear on the page - do not paraphrase, summarize, or change any wording
- Do not add any content that is not explicitly visible on the page
- Do not add explanatory text, headers, or commentary
- Preserve the exact spelling, capitalization, and punctuation as shown
- Use markdown formatting (# ## ### for headings, ** for bold, * for bullets, etc.) to match the visual hierarchy
- Maintain the exact order and structure of information as it appears
- If text is in a table, recreate the exact table structure in markdown
- Output ONLY the converted text - no introduction, no conclusion, no extra explanations

Convert this PDF page to markdown using the exact text shown:

{page_text}"""

        response = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=8192,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()

    def _extract_with_llm(self, text: str, file_path: Path, use_vision: bool = False) -> str:
        """Process text through LLM, using vision if specified."""
        try:
            # Split text by page markers to process individually
            pages = text.split("--- Page ")
            cleaned_pages = []

            for i, page_text in enumerate(pages):
                if not page_text.strip():
                    continue

                # Extract page number from the page marker
                actual_page_num = 1  # default
                if i > 0:
                    page_text = "--- Page " + page_text
                    # Extract page number from marker like "--- Page 1 ---"
                    page_match = re.search(r'--- Page (\d+)', page_text)
                    if page_match:
                        actual_page_num = int(page_match.group(1))

                if use_vision:
                    logger.info(f"Processing page {actual_page_num} with vision-enhanced LLM")
                    try:
                        cleaned_page = self._extract_with_vision(page_text, file_path, actual_page_num)
                        if cleaned_page:
                            cleaned_pages.append(cleaned_page)
                    except Exception as vision_error:
                        logger.warning(f"Vision processing failed for page {actual_page_num}, falling back to text-only: {vision_error}")
                        # Fallback to text-only even if vision was requested
                        cleaned_page = self._extract_text_only(page_text)
                        if cleaned_page:
                            cleaned_pages.append(cleaned_page)
                else:
                    logger.info(f"Processing page {actual_page_num} with text-only LLM")
                    cleaned_page = self._extract_text_only(page_text)
                    if cleaned_page:
                        cleaned_pages.append(cleaned_page)

            return "\n\n".join(cleaned_pages)

        except Exception as e:
            raise Exception(f"LLM processing failed: {e}")

    def _is_text_garbled(self, text: str) -> bool:
        """Check if extracted text appears garbled."""
        if not text or len(text.strip()) < 10:
            return True

        # Check for common garbling indicators from the problematic PDF
        garbled_indicators = [
            'UU',          # Multiple U's (spaces replaced with U)
            '(cid:',        # Character ID references
            'U3U',         # Specific pattern we saw
            'UandU',       # Another pattern we saw
            'RnderstandingU', # Pattern from garbled text
            '3hristF',     # Pattern from garbled text
        ]

        garbled_count = sum(1 for indicator in garbled_indicators if indicator in text)

        # Check for excessive character replacement patterns
        # Look for sequences like "6" where "y" should be
        weird_number_patterns = len(re.findall(r'[0-9](?=[a-z])|(?<=[a-z])[0-9]', text))

        # Check for excessive single character substitutions
        u_substitutions = text.count('U') - text.count('URL') - text.count('USA')  # Exclude legitimate uses

        # Calculate garbling score
        total_chars = len(text)
        if total_chars == 0:
            return True

        # Scoring system
        garbled_score = (
            garbled_count * 50 +           # Heavily weight known garbled patterns
            weird_number_patterns * 2 +    # Weight strange number placements
            max(0, u_substitutions - 10) * 1  # Weight excessive U's (allow some legitimate use)
        )

        # If garbling score is more than 1.5% of total characters, consider it garbled
        garbled_ratio = garbled_score / total_chars

        is_garbled = garbled_ratio > 0.015

        if is_garbled:
            logger.debug(f"Text appears garbled (score: {garbled_score}, ratio: {garbled_ratio:.4f})")

        return is_garbled