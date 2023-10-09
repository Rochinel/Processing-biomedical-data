# -*- coding: utf-8 -*-

"""
Created on fri sep 29/09/2023 00:18:35

@author: Nanfa Rochinel
"""

import pandas as pd
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
from typing import Any, List, Dict


def load_model(model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, compute: str = 'no-gpu') -> Any:
    """
    Load a Named Entity Recognition (biomedical-ner-all) model pipeline.

    This function loads a pre-trained biomedical-ner-all model pipeline with specified configurations.

    Args:
    model (AutoModelForTokenClassification): A pre-trained model for token classification (NER).
    tokenizer (AutoTokenizer): A pre-trained tokenizer compatible with the model.
    compute (str): A string specifying the compute device. Use 'gpu' for GPU or any other value for CPU.

    Returns:
    Any: A pipeline for NER processing.

    Example:
    # Load a pre-trained NER model pipeline with a GPU (if available)
    ner_model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
    ner_tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    ner_pipeline_gpu = load_model(ner_model, ner_tokenizer, 'gpu')

    # Load a pre-trained NER model pipeline on CPU
    ner_pipeline_cpu = load_model(ner_model, ner_tokenizer, 'cpu')
    """
    if compute == 'gpu':
        pipe = pipeline("ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="max",
                        device=0)
    else:
        pipe = pipeline("ner",
                        model=model,
                        tokenizer=tokenizer,
                        aggregation_strategy="max")

    return pipe


def load_deseases(file: str) -> List:
    df = pd.read_csv(file)
    entities = list(df['entities'])

    return entities


def ner_prediction(pipe: pipeline, corpus: str) -> pd.DataFrame:
    """
    Perform Named Entity Recognition (NER) prediction on a given corpus using a pre-trained model.

    This function takes a pre-trained NER pipeline, processes the provided corpus, and returns the NER results
    in a DataFrame containing columns for 'entity_group', 'word', and 'score'.

    Args:
    pipe (pipeline): A pre-trained NER pipeline from Hugging Face Transformers.
    corpus (str): The text corpus on which NER prediction is performed.

    Returns:
    pd.DataFrame: A DataFrame containing NER results with columns 'entity_group', 'word', and 'score'.

    Example:
    # Load a pre-trained NER pipeline
    ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", tokenizer="token-classification")

    # Perform NER prediction on a sample corpus
    text_corpus = "John Smith works at Google Inc. in New York City."
    ner_result_df = ner_prediction(ner_pipeline, text_corpus)
    print(ner_result_df)
    """
    # Perform NER prediction on the corpus
    result: List[Dict[str, Any]] = pipe(corpus)

    # Load the NER results into a DataFrame
    df = pd.DataFrame(result)

    # Select and return specific columns
    return df[['entity_group', 'word', 'start',	'end' , 'score']]
