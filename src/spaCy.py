# Install: pip install spacy && python -m spacy download en

import spacy
import pprint
from nltk import Tree


class SpaCy:

    def __init__(self, language='en'):
        # Load English tokenizer, tagger, parser, NER and word vectors
        self.nlp = spacy.load(language)

    def find_entities(self,text):
        doc = self.nlp(text)
        return doc.ents

    """
    # The structure has the shape of     dict<entities <dict<label array<doc_id> >>    [entity.text][entity.label_].append(id)
    def entity_dic(self, text, id):

        # Process whole documents
        #text = open(text).read()
        entities = {}
        doc = self.nlp(text)
        # Find named entities, phrases and concepts
        # new
        for entity in doc.ents:
            if (entity.text == " ") or (entity.text == "'s"):
                pass
            elif entity.text in entities:
                if entity.label_ in entities[entity.text]:
                    if id not in entities[entity.text]:
                        entities[entity.text][entity.label_].append(id)
                        #entities[entity.text].append(id)
                else:
                    entities[entity.text][entity.label_] = []
                    entities[entity.text][entity.label_].append(id)

            else:
                entities[entity.text] = {}
                entities[entity.text][entity.label_] = []
                entities[entity.text][entity.label_].append(id)

        return entities
    """
    def part_of_speech_tagging(self, text):
        doc = self.nlp(text)
        for token in doc:
            pprint.pprint(len(token.children))
            #print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,token.shape_, token.is_alpha, token.is_stop)
