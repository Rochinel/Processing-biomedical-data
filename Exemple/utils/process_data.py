# -*- coding: utf-8 -*-

"""
Created on fri sep 29/09/2023 00:18:35

@author: Nanfa Rochinel
"""
from typing import List, Dict, Any

import pandas as pd
from transformers import pipeline

from utils.load import ner_prediction


def summarize_entities(dataframe: pd.DataFrame, entity_list: List[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Summarize entity information from a DataFrame.

    This function takes a list of entity groups and a DataFrame containing entity data
    and returns a dictionary summarizing the information for each entity group.

    Args:
    entity_list (List[str]): List of entity groups to summarize.
    dataframe (pd.DataFrame): DataFrame containing entity data with columns 'entity_group', 'value', and 'score'.

    Returns:
    Dict[str, Dict[str, Dict[str, Any]]]: A dictionary summarizing entity information.

    Example:
    csv_data = '''
    entity_group,value,score
    Age,63 year old,0.998951256275177
    Sex,woman,0.9997994303703308
    History,no known known cardiac history,0.999748706817627
    Clinical_event,presented,0.9997915625572205
    Detailed_description,sudden onset,0.9998956322669983
    Sign_symptom,dyspnea,0.9999427795410156
    Therapeutic_procedure,intubation,0.9999328851699829
    Therapeutic_procedure,ventilatory support,0.999833345413208
    Biological_structure,chest,0.9998952150344849
    Sign_symptom,syncope,0.9997344613075256
    Sign_symptom,afebrile,0.9999532699584961
    Detailed_description,sinus,0.9999653100967407
    Sign_symptom,tachycardia,0.9999632835388184
    Lab_value,140 beats/min,0.9998094439506531
    '''

    df = pd.read_csv(StringIO(csv_data))
    entities_to_summarize = ["Age", "Sex", "Sign_symptom", "Lab_value"]
    result = summarize_entities(entities_to_summarize, df)
    print(result)
    """
    entity_summary: Dict[str, Dict[str, Any]] = {}

    if not entity_list:  # check if entity_list is empty
        unique_entities = dataframe["entity_group"].unique()
        for entity_group in unique_entities:
            entity_group_data = {"count": 0, "values": {}}

            entity_df = dataframe[dataframe["entity_group"] == entity_group]
            unique_values = entity_df["word"].unique()

            for value in unique_values:
                count = entity_df[entity_df["word"] == value].shape[0]
                entity_group_data["values"][value] = {}
                entity_group_data["values"][value]['count'] = count
                entity_group_data["values"][value]["start"] = entity_df[entity_df["word"] == value]["start"].values[0]
                entity_group_data["values"][value]["end"] = entity_df[entity_df["word"] == value]["end"].values[0]

                entity_group_data["count"] += count

            entity_summary[entity_group] = entity_group_data
    else:
        for entity_group in entity_list:
            entity_group_data: Dict[str, Any] = {"count": 0, "values": {}}

            # Filter the DataFrame to get only rows for the current entity group
            entity_df = dataframe[dataframe["entity_group"] == entity_group]

            # Get unique values within the entity group
            unique_values = entity_df["word"].unique()

            for value in unique_values:
                # Count the occurrences of the word in the filtered DataFrame
                count = entity_df[entity_df["word"] == value].shape[0]
                entity_group_data['values'][value] = {}
                entity_group_data["values"][value]['count'] = count
                # Add index on each value
                entity_group_data["values"][value]["start"] = entity_df[entity_df["word"] == value]["start"].values[0]
                entity_group_data["values"][value]["end"] = entity_df[entity_df["word"] == value]["end"].values[0]

                # Update the total count for the entity group
                entity_group_data["count"] += count

            # Store the entity group data in the summary dictionary
            entity_summary[entity_group] = entity_group_data

    return entity_summary


def corpus_analyzer(pipe: pipeline, corpus: str, entities: List[str] = None, compute: str = "no-gpu") -> Dict:
    """
    Analyze a corpus for Named Entity Recognition (NER) and summarize the results.

    This function takes a pre-trained NER pipeline, a corpus of text, and an optional list of entity groups to focus on.
    It performs NER prediction on the corpus, summarizes the NER results, and returns a dictionary containing the summary.

    Args:
    pipe (pipeline): A pre-trained NER pipeline from Hugging Face Transformers.
    corpus (str): The text corpus to analyze for NER.
    entities (List[str]): An optional list of entity groups to focus on. Default is an empty list (analyze all entities).
    compute (str): A string specifying the compute device. Use 'gpu' for GPU or any other value for CPU. Default is 'no-gpu'.

    Returns:
    Dict: A dictionary summarizing NER results for specified entity groups.

    Example:
    .. code-block:: python
    # Load a pre-trained NER pipeline
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")
    ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    # Analyze a sample corpus with a focus on specific entity groups
    text_corpus = "John Smith works at Google Inc. in New York City."
    focused_entities = ["Age", "Sex", "Sign_symptom", "Lab_value","Disease_disorder"]
    analysis_result = corpus_analyzer(ner_pipeline, text_corpus, focused_entities)
    print(analysis_result)

    # Analyze a corpus without specifying specific entity groups
    full_analysis_result = corpus_analyzer(ner_pipeline, text_corpus)
    print(full_analysis_result)
    """
    # Perform NER prediction on the corpus
    pred_df = ner_prediction(pipe=pipe, corpus=corpus)

    # Summarize NER results for specified entity groups or all entities
    summary = summarize_entities(entity_list=entities, dataframe=pred_df)

    return summary
