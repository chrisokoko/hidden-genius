# Document Processing Service

A simple, unified service that converts any document (audio, PDF, DOCX) to text.

## Features

- **Automatic file type detection** - Just provide a file, the service figures out how to process it
- **Multiple format support**:
  - Audio files (.m4a, .mp3, .wav, .ogg, .flac) - Transcription using Whisper or Google Speech
  - PDF files (.pdf) - Text extraction with PyPDF2
  - Word documents (.docx, .doc) - Text extraction with python-docx
- **Batch processing** - Process entire directories of mixed file types
- **Simple interface** - Just two arguments: input and output directory

## Installation

```bash
# Clone the repository
git clone [repository-url]

# Install dependencies
pip install -r document_processing_service/requirements.txt
```

## Usage

### Command Line

```bash
# Process a single file
python -m document_processing_service.cli document.pdf output/

# Process a directory of files
python -m document_processing_service.cli documents/ output/
```

### Python API

```python
from document_processing_service.core.factory import DocumentProcessor
from pathlib import Path

# Initialize processor
processor = DocumentProcessor()

# Process single file
result = processor.process_file(
    Path("document.pdf"),
    Path("output/")
)

# Process directory
results = processor.process_directory(
    Path("documents/"),
    Path("output/")
)
```

## Architecture

The service uses a **Factory Pattern** combined with **Strategy Pattern** for extensibility:

```
document_processing_service/
├── core/
│   ├── base.py              # Abstract base processor class
│   ├── types.py             # Shared types and data classes
│   └── factory.py           # Factory for processor selection
├── processors/
│   ├── audio.py             # Audio transcription processor
│   ├── pdf.py               # PDF text extraction processor
│   └── docx.py              # DOCX text extraction processor
├── utils/
│   ├── file_detection.py    # File type detection utilities
│   └── logging.py           # Logging configuration
└── cli.py                   # Command-line interface
```

### Key Components

1. **BaseProcessor** (`core/base.py`): Abstract base class that all processors implement
2. **DocumentProcessor** (`core/factory.py`): Factory class that selects the appropriate processor
3. **Processors** (`processors/`): Specific implementations for each file type
4. **CLI** (`cli.py`): Simple command-line interface using argparse

### Processing Flow

1. User provides input file/directory and output directory
2. Factory determines file type based on extension
3. Appropriate processor is selected and instantiated
4. File is processed and text is extracted
5. Text is saved to output directory with same name but .txt extension

## Output

All files are converted to plain text (.txt) files in the specified output directory:
- `document.pdf` → `output/document.txt`
- `audio.m4a` → `output/audio.txt`
- `report.docx` → `output/report.txt`

## Error Handling

- Unsupported file types are skipped with a message
- Failed processing is reported but doesn't stop batch processing
- Summary statistics are shown after batch processing

## Extending the Service

To add support for a new file type:

1. Create a new processor in `processors/` that extends `BaseProcessor`
2. Implement `can_process()` and `process()` methods
3. Add the processor to the factory's processor list in `core/factory.py`
4. Add any new dependencies to `requirements.txt`

## Requirements

- Python 3.8+
- ffmpeg (for audio processing)
- See `requirements.txt` for Python dependencies

## License

[Your license here]

 How to Use:

  # Install dependencies
  pip install -r document_processing_service/requirements.txt

  # Process single file
  python -m document_processing_service.cli document.pdf output/

  # Process directory
  python -m document_processing_service.cli documents/ output/

  # With verbose logging
  python -m document_processing_service.cli audio.m4a transcripts/ -v