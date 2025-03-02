# HTML Text Extractor

This application uses Google's Gemini AI to intelligently extract and process English text content from HTML sources. It can process both URLs and HTML files, making it ideal for content extraction and preprocessing for other APIs.

## Features

- Extract English text from HTML content using Gemini AI
- Process URLs directly
- Process local HTML files
- Intelligent text cleaning and formatting
- Easy integration with other APIs

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
4. Add your API keys to the `.env` file:
   - Add your Google Gemini API key (get one from https://makersuite.google.com/app/apikey)
   - Add any other API keys needed for your target API

## Usage

### Basic Usage

```python
from html_text_extractor import HTMLTextExtractor

extractor = HTMLTextExtractor()

# Process a URL
text = extractor.process_url("https://example.com")
print(text)

# Process an HTML file
text = extractor.process_html_file("path/to/file.html")
print(text)
```

### Integration with Other APIs

The extracted text can be easily passed to other APIs. Simply take the output from the extractor and use it as input for your target API.

## Requirements

- Python 3.7+
- Google Gemini API key
- Internet connection for URL processing
- Additional API keys as needed for your target APIs

## Error Handling

The application includes built-in error handling for:
- Invalid URLs
- File reading errors
- API communication issues
- HTML parsing errors

## License

This project is licensed under the MIT License - see the LICENSE file for details. 