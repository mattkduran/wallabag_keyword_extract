# Network Keyword Extractor

A Python-based tool for extracting meaningful keywords from text content and analyzing their relationships in a network graph structure. This tool is specifically designed for processing content from Wallabag entries but can be adapted for other text analysis needs.

## Features

- Intelligent keyword extraction with technical term boosting
- Multi-word phrase detection
- Network relationship analysis between keywords
- HTML content cleaning
- SQL generation for database integration
- Network visualization data generation

## Requirements

- Python 3.x
- pandas
- beautifulsoup4
- nltk
- difflib

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd network-keyword-extractor
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Usage

### Basic Usage

```python
from main import NetworkKeywordExtractor

extractor = NetworkKeywordExtractor()
keywords = extractor.extract_keywords("Your text here")
network_data = extractor.generate_network_data()
```

### Processing CSV Files

The tool can process CSV files containing Wallabag entries:

```python
from main import process_entries

results, network_data = process_entries("your_csv_file.csv")
```

### Key Components

#### NetworkKeywordExtractor Class

- `extract_keywords(text, max_keywords=5)`: Extracts most relevant keywords from text
- `generate_network_data()`: Creates network visualization data
- `clean_html(html_content)`: Cleans HTML content and extracts plain text

#### Text Analysis Features

- Technical term boosting for relevant keywords
- Phrase extraction for multi-word concepts
- Keyword merging for similar terms
- Stop word filtering
- Frequency analysis

#### Network Analysis

The tool generates network data with:
- Nodes: Keywords with frequency and connection information
- Edges: Relationships between keywords with weight calculations

## Output Files

The tool generates two main output files:

1. `tag_insertions.sql`: SQL statements for database integration
2. `tag_network.json`: Network visualization data in JSON format

## Network Data Structure

The network data JSON follows this structure:

```json
{
  "nodes": [
    {
      "id": "keyword",
      "frequency": 1,
      "connections": 4
    }
  ],
  "edges": [
    {
      "source": "keyword1",
      "target": "keyword2",
      "weight": 1
    }
  ]
}
```

## Algorithm Details

### Keyword Extraction

1. Text Preprocessing
   - HTML cleaning
   - Tokenization
   - Stop word removal

2. Scoring System
   - Base frequency score
   - Technical term boost (1.5x)
   - Specialty term boost (1.3x)
   - Length boost
   - Network presence boost (1.2x)

### Network Analysis

- Tracks keyword co-occurrence
- Calculates connection strengths
- Generates visualization-ready data

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
