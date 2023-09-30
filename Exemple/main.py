import pandas as pd
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification


from utils.load import load_model
from utils.process_data import corpus_analyzer





def main():

    # load how data
    data = ("A 63-year-old woman with no known cardiac history presented with a sudden onset of dyspnea requiring intubation and ventilatory support out of hospital. "
            "She denied preceding symptoms of chest discomfort, palpitations, syncope or infection. "
            "The patient was afebrile and normotensive, with a sinus tachycardia of 140 beats/min, corona, malaria")

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
    model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

    # use how custome package utils to load model into pipe and process the result of our prediction.
    pipe = load_model(model=model, tokenizer=tokenizer)

    # lets analyse our corpus now
    summary = corpus_analyzer(pipe=pipe, corpus=data, entities=entities)

    print('Corpus_Analyser' : summary)



if __name__ == "__main__":
    main()