import numpy as np
import re
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns

def _preprocess_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = text.lower()
    tokens = text.split()
    return tokens

def _compute_cooccurrence(tokens, entities, window_size):
    matrix = np.zeros((len(entities), len(entities)))
    for i in range(len(entities)):
      for j in range(len(entities)):
        if i == j :
          matrix[i][j] = 1



    for i, token in enumerate(tokens):
        if token in entities:
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            window = tokens[start:end]

            for other_token in window:
                if other_token in entities and other_token != token:
                    matrix[entities.index(token)][entities.index(other_token)] += 1

    return matrix


def _apply_threshold(matrix, threshold):
    matrix[matrix < threshold] = 0
    return matrix

def cooccurrence_analysis(text, data_dict, window_size=30, threshold=0.05):
    """
    Analyzes the co-occurrence of entities in a given text based on a predefined data dictionary.

    Args:
    text (str): The input text where entities will be searched and analyzed.
    data_dict (dict): A dictionary containing entities and their information.
                      It is expected to have a specific structure where each entity type
                      (e.g., 'Disease', 'Symptom') is a key, and the value is another dictionary
                      with 'count' and 'values' as keys.
    window_size (int, optional): The size of the sliding window that will be used to find co-occurring entities.
                                 It defines the maximum distance between two entities to consider them as co-occurring.
                                 Defaults to 5.
    threshold (float, optional): The threshold to apply after normalizing the co-occurrence matrix.
                                 Any value below this threshold will be set to 0. Defaults to 0.1.

    Returns:
    tuple: A tuple containing the normalized and thresholded co-occurrence matrix and the list of entities.
           The co-occurrence matrix is a 2D NumPy array where each row and column corresponds to an entity,
           and the value at matrix[i][j] represents the normalized co-occurrence frequency of entity i with entity j.
           The list of entities is a list of strings, where each string is an entity found in the text.
    """
    tokens = _preprocess_text(text)
    entities = [entity for entity_type in data_dict.values() for entity in entity_type['values']]
    matrix = _compute_cooccurrence(tokens, entities, window_size)
    diagonal = np.diag(matrix).copy()
    matrix = normalize(matrix, axis=1, norm='l1')
    np.fill_diagonal(matrix, diagonal)
    matrix = _apply_threshold(matrix, threshold)
    return matrix, entities



def plot_cooccurrence_matrix(matrix, entities):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=entities, yticklabels=entities)
    plt.title("Matrice de Co-occurrence")
    plt.show()