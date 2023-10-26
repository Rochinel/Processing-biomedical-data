from typing import List

import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


from utils.load import load_model
from utils.process_data import corpus_analyzer
from utils.co_occurence_matrix import cooccurrence_analysis, plot_cooccurrence_matrix





def main():

    # load how data
    data = ("A 63-year-old woman with no known cardiac history presented with a sudden onset of dyspnea requiring intubation and ventilatory support out of hospital. "
            "She denied preceding symptoms of chest discomfort, palpitations, syncope or infection. "
            "The patient was afebrile and normotensive, with a sinus tachycardia of 140 beats/min, corona, malaria")

    doc = """
    	CASE: A 28-year-old previously healthy man presented with a 6-week history of palpitations. 
          The symptoms occurred during rest, 2–3 times per week, lasted up to 30 minutes at a time 
          and were associated with dyspnea. Except for a grade 2/6 holosystolic tricuspid regurgitation 
          murmur (best heard at the left sternal border with inspiratory accentuation), physical 
          examination yielded unremarkable findings.
          """



    # list of entities we want to detecte
    entities: list[str] = ["Age", "Sex", "Sign_symptom", "Lab_value", "Disease_disorder"]

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # use how custome package utils to load model into pipe and process the result of our prediction.
    pipe = load_model(model=model, tokenizer=tokenizer)

    # lets analyse our corpus now
    summary = corpus_analyzer(pipe=pipe, corpus=data)
    summary_2 = corpus_analyzer(pipe=pipe, corpus=doc)

    print('Corpus_Analyser_1: ', summary)
    print('corpus_Analyse_2: ', summary_2)


    # Generate Co_corrunce matrix for data.

    matrix, entities = cooccurrence_analysis(data, summary, window_size=30, threshold=0.05)

    # plot the matrix

    plot_cooccurrence_matrix(matrix, entities)


if __name__ == "__main__":
    main()