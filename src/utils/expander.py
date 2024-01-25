import json
import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


class AcronymExpander:
  def __init__(self, nlp, diction, model_name, device='cpu'):
    self.nlp = nlp
    self.diction = diction
    self.device = device
    self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def softmax_scaler(self, probs):
    values = list(probs.values())
    softmax_probs = np.exp(values)/np.sum(np.exp(values))
    return dict(zip(probs.keys(), softmax_probs))

  def get_acronyms_from_sentence(self, sentence):
    return [token.text for token in self.nlp(sentence) if token.text in self.diction.keys()]

  def get_true_probs(self, acronym, expansion, sentence):
    text = self.tokenizer.sep_token.join([acronym, expansion, sentence])
    inputs = self.tokenizer.encode_plus(text, truncation=True, return_tensors='pt').to(self.device)
    with torch.no_grad():
      outputs = self.model(**inputs)
    logits = outputs.logits
    probs_tensor = torch.softmax(logits, dim=1)
    true_prob = probs_tensor[:, 1].item()
    return true_prob

  def get_expansions_probs(self, sentence, acronym):
    probs = {}
    expansions = self.diction[acronym]
    for expansion in expansions:
      true_prob = self.get_true_probs(acronym, expansion, sentence)
      probs[expansion] = true_prob
    probs = self.softmax_scaler(probs)
    return probs

  def predict(self, sentence, acronym):
    probs = self.get_expansions_probs(sentence, acronym)
    expansion = max(probs, key=probs.get)
    return expansion, probs

  def format_prediction(self, sentence, acronym):
    expansion, probs = self.predict(sentence, acronym)
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    res = {
        "sentence": sentence,
        "acronym": acronym,
        "expansion": expansion,
        "probs": [{"expansion":expansion, "prob":prob} for expansion, prob in sorted_probs]
    }
    return res
