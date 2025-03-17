# TF-IDF Search Engine with Inverted Index

A powerful and flexible TF-IDF-based search engine that integrates advanced text preprocessing with an efficient inverted index architecture for information retrieval tasks.

## Features

- **Text Preprocessing Pipeline**: Comprehensive preprocessing with customizable components
  - Lowercase conversion
  - Diacritics removal
  - Stop words removal
  - Nonsense token filtering
  - Stemming (with automatic detection and fallback)

- **TF-IDF Computation**: Industry-standard term weighting
  - Term Frequency (TF) calculation: 1 + log10(frequency)
  - Inverse Document Frequency (IDF): log10(N/df)
  - Vector space model for document representation

- **Inverted Index**: Efficient document retrieval
  - Word-to-document mapping for fast lookups
  - Frequency preservation for accurate scoring

- **Cosine Similarity Ranking**: Relevance scoring based on vector similarity
  - Normalization for documents of different lengths
  - Fast computation with sparse vector optimization

## Installation

1. Clone this repository:
```bash
git clone https://github.com/VandlJ/tf-idf-search.git
cd tf-idf-search
```

2. (Optional) Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Ensure the stemming tool is properly set up (if using stemming):
```bash
# For Czech language stemming with Snowball stemmer
cd preprocessing/CzechSnowballStemmer
make
chmod +x stemmer
```

## Usage

### Basic Search

Run a search using test documents and default queries:
```bash
python3 tfidf_search.py
```

### Custom Queries

Specify your own search queries:
```bash
python3 tfidf_search.py --queries "electric vehicles" "volkswagen cars" "hybrid technology"
```

### Using Your Own Document Collection

Search through your own JSON document collection:
```bash
python3 tfidf_search.py --input preprocessing/data/autorevue.json --queries "volkswagen" "vozy cupra"
```

### Adjust Number of Results

Control how many top results are displayed:
```bash
python3 tfidf_search.py --top 5
```

### Debug Mode

Use verbose or debug mode to see detailed information about the search process:
```bash
python3 tfidf_search.py --verbose
python3 tfidf_search.py --debug
```

## JSON Document Format

The input JSON file should contain an array of documents, each with the following structure:
```json
[
  {
    "title": "Document Title",
    "abstract": "A brief summary of the document",
    "text": "The full text content of the document..."
    ...
  },
  ...
]
```

## Core Functions

### Document Preprocessing

- `preprocess_text(text, pipeline)`: Processes text with the given preprocessing pipeline
- `create_preprocessing_pipeline()`: Creates a standard preprocessing pipeline with components like stemming

### TF-IDF Calculation

- `compute_tf(word_freq)`: Calculates term frequency scores
- `compute_tf_idf_vectors(inverted_index)`: Computes TF-IDF vectors for all documents
- `process_queries(queries, pipeline, inverted_index)`: Processes queries and calculates their TF-IDF vectors

### Document Retrieval

- `load_documents(file_path)`: Loads documents from a JSON file
- `process_documents(documents_json, pipeline)`: Processes documents and builds the inverted index
- `rank_documents(query_vector, doc_vectors, documents, top_k)`: Ranks documents by similarity to query

### Similarity Calculation

- `compute_cosine_similarity(vec1, vec2)`: Calculates the cosine similarity between two vectors

## System Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌───────────────┐
│ Document Loader ├────►│ Preprocessing ├────►│ Inverted Index│
└─────────────────┘     └───────────────┘     └───────┬───────┘
                                                      │
                                                      ▼
┌─────────────────┐     ┌───────────────┐     ┌───────────────┐
│ Query Processing◄─────┤ Ranking Engine◄─────┤  TF-IDF Model │
└─────────┬───────┘     └───────────────┘     └───────────────┘
          │
          ▼
┌─────────────────┐
│  Search Results │
└─────────────────┘
```

## (Default) Preprocessing Pipeline Components

1. **LowercasePreprocessor**: Converts text to lowercase
2. **RemoveDiacriticsPreprocessor**: Removes diacritical marks
3. **StopWordsPreprocessor**: Filters out common stop words
4. **NonsenseTokenPreprocessor**: Removes tokens below a specified length
5. **StemPreprocessor**: Reduces words to their stems

## Advanced Usage

### Customizing the Pipeline

You can create a custom preprocessing pipeline by modifying the `create_preprocessing_pipeline()` function:
```py
def create_custom_pipeline():
    preprocessors = [
        LowercasePreprocessor(),
        RemoveDiacriticsPreprocessor(),
        # Add or remove preprocessors as needed
    ]
    return PreprocessingPipeline(preprocessors, name="Custom Pipeline")
```

### Working with Large Document Collections

For large document collections, consider:

   1. Batching document processing
   2. Storing the inverted index on disk
   3. Using multiprocessing for parallel computation

## Example

```bash
# Run a search for "volkswagen electric cars" on test documents
python3 tfidf_search.py --input preprocessing/data/autorevue.json --queries "volkswagen" "vozy cupra" --verbose
```

Sample output: 
```
==================================================
SEARCH RESULTS
==================================================

Query 1: "volkswagen"
Processed query terms: ['volkswag']
----------------------------------------
1. Document: "Nejvýkonnější Volkswagen ID.Buzz má český ceník. Verzi GTX můžete objednávat již nyní"
   Abstract: Výkon 340 koní, pohon všech kol, dobrá tažná síla i solidní dojezd – ostrá verze vstupuje do prodeje...
   Similarity: 0.0772
   Matching terms: volkswag

2. Document: "Nejlevnější nový Volkswagen Tayron vypadá takto. Plug-in jen pro pět, zato se skvělým dojezdem"
   Abstract: Nové rodinné SUV míří do prodeje se zatím omezenou nabídkou výbav i motorizací. Základní specifikace...
   Similarity: 0.0707
   Matching terms: volkswag

3. Document: "Nový Volkswagen Tayron oficiálně: Rival Kodiaqu má sedm míst i tradiční motory"
   Abstract: Německá automobilka vrací do hry přímého rivala Škody Kodiaq, tentokrát jde však o nový samostatný m...
   Similarity: 0.0530
   Matching terms: volkswag


Query 2: "vozy cupra"
Processed query terms: ['vozit', 'cup']
----------------------------------------
1. Document: "Taková garáž v Česku ještě nebyla. Je plná vozů Cupra a nedaleko centra Prahy"
   Abstract: Pro vozy Cupra se chodí do garáže. A od minulého týdne se pro ně můžete vydat i do Středočeského kra...
   Similarity: 0.2191
   Matching terms: vozit, cup

2. Document: "Nová Škoda Superb Sportline vstupuje na český trh. I s jedna-pětkou TSI stojí skoro 1,1 milionu"
   Abstract: Akčně laděná verze jde na trh jako liftback i kombi, pod kapotou může mít i plug-in hybrid.
   Similarity: 0.0060
   Matching terms: vozit

3. Document: "Toyota v Česku vyrazila na roadshow s auty pro profíky. Veze kompletní nabídku včetně přestaveb!"
   Abstract: Vyrazili jsme na jednu ze zastávek Toyota Professional Roadshow, která letos nabízí pestrý program p...
   Similarity: 0.0058
   Matching terms: vozit
```

## Author

VandlJ