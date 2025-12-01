# yh_rag_cloud_api/parsers/docx_utils.py

import zipfile
import xml.etree.ElementTree as ET
from io import BytesIO
from typing import Optional


def extract_text_from_docx_bytes(docx_bytes: bytes) -> Optional[str]:
    """
    Extract text from DOCX bytes using multiple reliable methods.
    Returns: Extracted text or None if all methods fail
    """
    # Method 1: Try python-docx (most reliable)
    try:
        import docx
        doc = docx.Document(BytesIO(docx_bytes))
        full_text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text.strip())
        return '\n'.join(full_text)
    except ImportError:
        print("python-docx not available")
    except Exception as e:
        print(f"python-docx extraction failed: {e}")

    # Method 2: Manual DOCX XML extraction
    try:
        with zipfile.ZipFile(BytesIO(docx_bytes)) as docx_zip:
            if 'word/document.xml' in docx_zip.namelist():
                with docx_zip.open('word/document.xml') as document_file:
                    tree = ET.parse(document_file)
                    root = tree.getroot()

                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    texts = []
                    for paragraph in root.findall('.//w:p', ns):
                        paragraph_text = []
                        for run in paragraph.findall('.//w:r', ns):
                            for text in run.findall('.//w:t', ns):
                                paragraph_text.append(text.text)
                        if paragraph_text:
                            texts.append(''.join(paragraph_text))
                    return '\n'.join(texts)
    except Exception as e:
        print(f"Manual DOCX extraction failed: {e}")

    return None