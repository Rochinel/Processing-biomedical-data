# Processing Biomedical Data

This repository provides tools and utilities for processing biomedical data, particularly focusing on Named Entity Recognition (NER) for biomedical entities.

## Directory Structure:

- **utils**: Contains utility functions for loading and processing data.
- **Data**: Contains data files used in the project.
- **Exemple**: Contains example scripts demonstrating the usage of the utilities.

## Utils Package:

The `utils` package is the core of this repository and contains the following modules:

### 1. **load.py**:
   - **Functions**:
     - `load_deseases`: Load diseases data.
     - `load_model`: Load the NER model.
     - `ner_prediction`: Perform NER prediction on the provided data.
   - **Dependencies**: This module uses libraries such as pandas, transformers, and typing.

### 2. **process_data.py**:
   - **Functions**:
     - `corpus_analyzer`: Analyze the corpus and provide insights.
     - `summarize_entities`: Summarize the detected entities from the NER predictions.
   - **Dependencies**: This module leverages pandas, transformers, and typing among others.

### 2. **co_occurence_matrix.py**
   - **Functions**:
     - `cooccurrence_analysis`: Analyze the corpus using the summarize _entities output and corpus to give the cooccurence matrice between the detected entities.
     - `plot_cooccurrence_matrix`: show the cooccurence matrix using the matplotlib librairy.
   - **Dependencies**: This module leverages numpy, re, scikit-learn, and typing among others.

## Usage:

To use the utilities provided in this repository, you can refer to the scripts in the `Exemple` directory which demonstrate how to load models, perform NER predictions, and analyze the results.

