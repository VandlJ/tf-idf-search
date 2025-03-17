# Text Preprocessing Configuration

This document describes configuration options in the `config.json` file.

## Input data (`input`)

| Key | Description |
|------|-------|
| `file` | Path to the input data file |
| `encoding` | Input file encoding (e.g. "utf-8") |

## Output data (`output`)

| Key | Description |
|------|-------|
| `directory` | Output directory for processed files |
| `original_vocab` | Filename for the original vocabulary |
| `preprocessed_vocab` | Filename for the preprocessed vocabulary |
| `stemmed_vocab` | Filename for the stemmed vocabulary |

## Preprocessing (`preprocessing`)

| Key | Description |
|------|-------|
| `lowercase` | Whether to convert text to lowercase (`true`/`false`) |
| `remove_diacritics` | Whether to remove diacritics (`true`/`false`) |

### Stop words (`stop_words`)

| Key | Description |
|------|-------|
| `use` | Whether to remove stop words (`true`/`false`) |
| `language` | Which stop words languages to use: `"czech"`, `"english"`, `"both"`, `"none"` |

### Nonsense tokens (`nonsense_tokens`)

| Key | Description |
|------|-------|
| `remove` | Whether to remove nonsense tokens (`true`/`false`) |
| `min_word_length` | Minimum word length to preserve (number) |

## Stemming

| Key | Description |
|------|-------|
| `use` | Whether to use stemming (`true`/`false`) |
| `language` | Language for stemming: `"cz"`, `"en"` |
| `stemmer_path` | Path to the stemmer binary |

## Lemmatization (`lemmatization`)

| Key | Description |
|------|-------|
| `use` | Whether to use lemmatization (`true`/`false`) |
| `language` | Language for lemmatization: `"cz"`, `"en"` |
| `model_path` | Path to the UDPipe model (null for automatic download) |
