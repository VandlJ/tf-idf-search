# Text Preprocessing Pipeline
This project provides a flexible text preprocessing pipeline for natural language processing (NLP) tasks. It supports tokenization, various preprocessing techniques, stemming, and lemmatization, with a focus on handling both Czech and English texts.

## Features
- **Tokenization**: Advanced regex-based tokenizer that identifies different token types (words, numbers, URLs, dates, HTML tags, punctuation)
- **Preprocessing options**:
  - Lowercase conversion
  - Diacritics removal
  - Stop words filtering (Czech, English, or both)
  - Filtering of nonsense tokens (by length and type)
- **Text normalization**:
  - Stemming via external Snowball stemmer (Czech/English)
  - Lemmatization via UDPipe (Czech/English)
- **Configurable pipeline**: JSON-based configuration with detailed documentation
- **Vocabulary statistics**: Track vocabulary size reduction through each step

## Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/text-preprocessing
cd text-preprocessing
```
2. Create and activate a virutal environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required dependencies:
```bash
pip3 install -r requirements.txt
```

## Usage 

1. Create a configuration file
```bash
python3 create_default_config.py
```
This creates two files:
- `config.json`: The configuration file to use with the program
- `CONFIG.md`: Documentation of all configuration options
2. Adjust the configuration

Edit `config.json` to match your requirements. Key options include:
- Input/output file paths
- Preprocessing options (lowercase, diacritics, stop words)
- Wheter to use stemming and/or lemmaization
- Language selectoin
3. Run the preprocessing pipeline
```bash
python3 main.py
```
4. View results
Results will be saved in the output directory (default: `results`) including:
- Original vocabulary with frequencies
- Preprocessed vocabulary
- Stemmed vocabulary (if enabled)
- Lemmatized vocabulary (if enabled)
- Groups of words sharing the same stem/lemma

## Project structure

- `main.py`: Main script that orchestrates the preprocessing pipeline
- `tokenizer.py`: Tokenization implementations
- `preprocess.py`: Preprocessing operations (lowercase, stop words, etc.)
- `stemming.py`: Interface to external stemming tools
- `lemmatization.py`: UDPipe-based lemmatization
- `create_default_config.py`: Configuration file generator

## Configuration Options

For detailed configuration options, see `CONFIG.md` generated alongside your config file. The main sections include:
- **Input data**: Source file and encoding
- **Output data**: Output directory and filenames
- **Preprocessing**: Text transformation options
- **Stemming**: Stemming configuration
- **Lemmatization**: UDPipe lemmatization settings

## Processing Order

The preprocessing steps are executed in the following order:
1. Stop words removal
2. Nonsense token filtering
3. Lowercase conversion
4. Diacritics removal
5. Stemming (if enabled)
6. Lemmatization (if enabled)

This order is designed to optimize the processing pipeline for most NLP tasks.

## Example

Starting with an input file of text documents:
```json
[
  {
    "title": "Example Article",
    "abstract": "This is a sample abstract.",
    "text": "This is the main content of the article with some example text."
  }
]
```

The pipeline:
1. Tokenizes the text
2. Applies the selected preprocessing steps
3. Optionally performs stemming and/or lemmatization
4. Generates vocabulary files with frequency counts

## Requirements

- Python 3.7 or higher
- External Snowball stemmer binary (for stemming)
- UDPipe library and models (for lemmatization)
- Input data in JSON format with "title", "abstract", and "text" fields
- Download tagger for lemmatization: `curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3131{/czech-pdt-ud-2.5-191206.udpipe}`
- Downlad and build stemmer from [URL](https://www.fit.vut.cz/research/product/133/.cs)
