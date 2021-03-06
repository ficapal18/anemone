import argparse
import cmd
import sys
import re
import os
import pickle

from nltk.tokenize import StanfordTokenizer


import pandas as pd
import src.spaCy as sp
from src.entity import Entity
from src.document import Document
from src.graph import GraphObject
from src.similarity import Similarity
from src.prompt import Prompt
from src.query import Query

#inv_map = {v: k for k, v in my_map.items()}
class Main:

    def __init__(self):

        self.entities = []
        self.counter_entities = 0
        self.ent2idx = {}
        self.load_equal_entities()

        self.documents = []
        self.counter_documents = 0
        self.amount_documents = 80000#1000

        self.queries = []
        self.counter_queries = 0

        self.spacy_instance = sp.SpaCy()
        self.similarity_object = Similarity()
        self.graph_instance = GraphObject()

        self.ent_doc_dataset_path = './data/ent_doc_dataset/ent_doc_dataset_path.pickle'

    def main(self, path_documents):

        self.document_files = self.load_documents(path_documents)

        #self.create_and_train_similarity(path_documents, similarity_object)

        #self.train_models(path_documents)


        self.similarity_object.load_dictionary()

        self.similarity_object.load_lda()
        self.similarity_object.load_lda_index()

        #self.show_topics(self.similarity_object.lda, 70, self.similarity_object.dictionary)


        #self.similarity_object.load_corpus()

        #self.similarity_object.load_tf_idf()
        #self.similarity_object.load_tf_idf_index()

        #self.similarity_object.tf_idf_similarity_matrix()
        #self.similarity_object.tf_idf_train()


        """
        import time
        import sent2vec


        sentences = ["first sentence .", "another sentence that I don't like so much, I'll go there Mr. Obama."]
        SNLP_TAGGER_JAR = os.path.join("./utils/stanford-postagger-full-2018-02-27/", "stanford-postagger.jar")
        tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
        s = ' <delimiter> '.join(sentences) #just a trick to make things faster
        tokenized_sentences_SNLP = self.tokenize_sentences(tknzr, [s])
        tokenized_sentences_SNLP = tokenized_sentences_SNLP[0].split(' <delimiter> ')
        assert(len(tokenized_sentences_SNLP) == len(sentences))
        print(tokenized_sentences_SNLP)

        model = sent2vec.Sent2vecModel()
        model.load_model('./data/models/wiki_unigrams.bin')
        emb = model.embed_sentence("once upon a time .")
        t0 = time.time()
        embs = model.embed_sentences(["first sentence .", "another sentence"])
        print(time.time() - t0)
        """

        #sys.exit()

        #for i in range(0, lda.num_topics - 1):
        #    print(lda.show_topic(i))
        #documents = self.load_documents(path_documents)
        #gen_docs = similarity_object.tokenize_documents(documents['body'])
        #similarity_object.generate_corpus(gen_docs)
        #similarity_object.lda_train()
        #self.similarity_object.tf_idf_train()

        self.fill_dataset_and_graph()

        # """ UNCOMMENT
        p = Prompt(self, sys.argv[1:])
        p.cmdloop()

        # """

    def fill_dataset_and_graph(self):
        print("Reading documents and finding entities.")
        # Fill our objects
        self.fill_ent_doc_memory_database(self.document_files, load=True)  # change to load or create

        # Fill our Neo4j Graph
        self.graph_instance.populate(self.entities, self.ent2idx, self.documents, self.queries)

        # If a document doesn't have entities it will not appear in the graph: NOT true anymore, because it will have similarities
        print("\tAdding entities and documents to the graph.")
        for idx, document in enumerate(self.documents):
            self.graph_instance.add_ent_to_doc(document)
            self.update_progress(idx+1, len(self.documents))
        print('\n')

        print("\tAdding similarities to the graph.")
        for idx, document in enumerate(self.documents):
            self.graph_instance.add_similarity_to_doc(document)
            self.update_progress(idx+1, len(self.documents))
        print('\n')

    def make_query(self, query_text):
        query_object = Query(self.counter_queries, query_text)
        entities_found = self.text_to_ent_idx(query_text)
        query_object.add_entities(entities_found)
        doc_similarities = self.find_similarities(query_text)
        query_object.add_similarities(doc_similarities)
        self.queries.append(query_object)
        self.counter_queries += 1
        self.graph_instance.add_ent_to_doc(query_object, type='QUERY')
        self.graph_instance.add_similarity_to_query(query_object, type='QUERY')

        print("SiMILARITEISASAS", query_object.similarities)

        """
        search = "Mr. Obama and Mr. Donald J. Trump both like to eat ice-creams"
        entities_found = self.text_to_ent_idx(search)
        response = graph_instance.find_documents_on_entities(entities_found)

        # print(response[0].get('labels'), response[0].get('name'))
        # print(response)
        """

    # The input format of documents should be a dataframe with columns; head & body.
    # This function is meant to fill the entities and documents.
    def fill_ent_doc_memory_database(self, documents, load=True):
        print("\tAdding entities, documents and similarities to our database.")
        if load:

            with open(self.ent_doc_dataset_path, 'rb') as handle:
                from_pickle = pickle.load(handle)
            self.entities= from_pickle["entities"]
            self.counter_entities = from_pickle["counter_entities"]
            self.documents = from_pickle["documents"]
            self.ent2idx = from_pickle["ent2idx"]
            self.documents = from_pickle["documents"]
            self.counter_documents = from_pickle["counter_documents"]
            self.amount_document = from_pickle["amount_documents"]

        else:
            if self.amount_documents > len(documents):
                self.amount_documents= len(documents)
            for idx_doc, document in documents.iterrows():

                if idx_doc >= self.amount_documents:
                    break
                import time
                #t0 = time.time()
                # entities
                doc_entities = self.fill_ent_database(self.counter_documents, document['body']+document['head'])
                document_object = Document(self.counter_documents, document['head'], document['body'])
                document_object.add_entities(doc_entities)
                #t1 = time.time() -t0
                #print("time entities:",t1 )
                #t0 = time.time()
                # similarities
                doc_similarities = self.find_similarities(document['body']+document['head'])
                document_object.add_similarities(doc_similarities)
                #print("time similarities:",  time.time() - t0)
                self.documents.append(document_object)
                self.counter_documents += 1
                self.update_progress(idx_doc+1, self.amount_documents)

            to_pickle= {
                "entities": self.entities,
                "counter_entities": self.counter_entities,
                "documents": self.documents,
                "ent2idx": self.ent2idx,
                "documents": self.documents,
                "counter_documents":self.counter_documents,
                "amount_documents": self.amount_documents
            }
            with open(self.ent_doc_dataset_path, 'wb') as handle:
                pickle.dump(to_pickle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("The dataset has been stored in", self.ent_doc_dataset_path)

    # Finds the entities of a text document and maintains coherence with the entities list.
    def fill_ent_database(self, doc_id, text):
        # Text(text) Start End Label(label_) Description
        entities = self.spacy_instance.find_entities(text)
        doc_entities = []
        for ent in entities:
            if self.is_valid_entity(ent):
                label = ent.text+'_'+ent.label_
                if (label) not in self.ent2idx:
                    self.ent2idx[label] = self.counter_entities
                    entity_object = Entity(self.counter_entities, ent.text, ent.label_)
                    entity_object.append_document(doc_id)
                    self.entities.append(entity_object)
                    self.counter_entities += 1
                    doc_entities.append(self.ent2idx[label])
                else:
                    if doc_id not in self.entities[self.ent2idx[label]].documents:
                        self.entities[self.ent2idx[label]].append_document(doc_id)
                        doc_entities.append(self.ent2idx[label])
        return doc_entities

    def find_similarities(self, text):
        return self.similarity_object.query_to_lda(text, language='english', k=20, verbose = False)

    # Finds the entities of a text document (??? I DONT KNOW and maintains coherence with the entities list.)
    def text_to_ent_idx(self, text):
        # Text(text) Start End Label(label_) Description
        entities = self.spacy_instance.find_entities(text)
        doc_entities = []
        total_ent_in_database = 0
        for ent in entities:
            if self.is_valid_entity(ent):
                label = ent.text+'_'+ent.label_
                # Check if its already there in case because it could have an alias
                if (label) in self.ent2idx:
                    entity = self.ent2idx[label]
                    #total_ent_in_database += 1
                    doc_entities.append(entity)
                else:
                    print("Entity <", label, ">not found in the graph")
        return doc_entities

    # There are some kind of buggy entities that we will filter out.
    def is_valid_entity(self,ent):
        text = ent.text
        label = ent.label_
        if ('_' in text) or ('—' in text) or (text == " ") or (text == "'s") or (text == " ") or (text == "  ") or (text == " ") or (text == "     "):
            return False
        elif (label == 'PERSON') or (label == 'ORG') or (label == 'LOC') or (label == 'DOCUMENT') or (label == 'NORP') or (label == 'GPE'):
            return True
        else: return False

    # This is to assign multiple names of entities to a single id, e.g: D.J.Trump and Trump
    def load_equal_entities(self):
        list = []
        equal1 = ['Donald J. Trump_PERSON', 'Trump_PERSON', 'Trump_ORG', 'Trump_NORP', 'Trump_GPE']
        equal2 = ['Queen Elizabeth II_PERSON', 'Queen Elizabeth_PERSON']
        equal3 = ['Obama_PERSON', 'Obama_GPE']
        list.append(equal1)
        list.append(equal2)
        list.append(equal3)
        for equality_group in list:
            for count, equality in enumerate(equality_group):
                ent_text, ent_label = equality.split('_')
                if count == 0:
                    entity_object = Entity(self.counter_entities, ent_text, ent_label)
                else:
                    entity_object.append_alias(ent_text)
                self.ent2idx[equality] = self.counter_entities
            self.entities.append(entity_object)
            self.counter_entities += 1

    def train_models(self, path_documents):
        documents = self.load_documents(path_documents)
        gen_docs = self.similarity_object.tokenize_documents(documents['body'])
        self.similarity_object.generate_corpus(gen_docs)
        self.similarity_object.lda_train()
        self.similarity_object.tf_idf_train()

    # Function to load the articles.csv dataset.
    def load_documents(self, path):
        with open(path) as myCSV:
            # idx, title, publication, author, date, year, month, url, content
            data = pd.read_csv(myCSV)
        documents = data[['title', 'content']].copy()
        documents.rename(columns={'title': 'head', 'content': 'body'}, inplace=True)
        return documents

    def update_progress(self, count, total):
        bar_len = 60
        suffix = ''
        filled_len = int(round(bar_len * count / float(total)))
        percents = round(100.0 * count / float(total), 1)
        bar = '=' * filled_len + '-' * (bar_len - filled_len)
        sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
        sys.stdout.flush()

    def create_and_train_similarity(self, path_documents, similarity_object):
        documents = self.load_documents(path_documents)
        gen_docs = similarity_object.tokenize_documents(documents['body'])
        similarity_object.generate_corpus(gen_docs)
        similarity_object.lda()

    def show_topics(self,model, num_topics, dict_sim):
        for i, topic in enumerate(model.show_topics(num_topics=num_topics, formatted=False)):
            tmp_int = 1
            p_total = 0
            first = True
            for id,p in topic[1]:
                p_total += p
                if p >= 0.009:
                    if first:
                        print("Topic #", topic[0], ":")
                        first = False
                    print(str(tmp_int),"(",str('{0:.2f}'.format(p)),"):\t",str(dict_sim[int(id)]))
                tmp_int += 1

    def tokenize(self, tknzr, sentence, to_lower=True):
        """Arguments:
            - tknzr: a tokenizer implementing the NLTK tokenizer interface
            - sentence: a string to be tokenized
            - to_lower: lowercasing or not
        """
        sentence = sentence.strip()
        sentence = ' '.join([self.format_token(x) for x in tknzr.tokenize(sentence)])
        if to_lower:
            sentence = sentence.lower()
        sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
        sentence = re.sub('(\@[^\s]+)','<user>',sentence) #replace @user268 by <user>
        filter(lambda word: ' ' not in word, sentence)
        return sentence

    def format_token(self, token):
        """"""
        if token == '-LRB-':
            token = '('
        elif token == '-RRB-':
            token = ')'
        elif token == '-RSB-':
            token = ']'
        elif token == '-LSB-':
            token = '['
        elif token == '-LCB-':
            token = '{'
        elif token == '-RCB-':
            token = '}'
        return token

    def tokenize_sentences(self, tknzr, sentences, to_lower=True):
        """Arguments:
            - tknzr: a tokenizer implementing the NLTK tokenizer interface
            - sentences: a list of sentences
            - to_lower: lowercasing or not
        """
        return [self.tokenize(tknzr, s, to_lower) for s in sentences]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anemone')
    parser.add_argument('-d','--documents', help='Path to documents dataset',
                        default='./data/documents/articles.csv')
    parser.add_argument("-l","--loaddefault", help="load existing database",
                        action="store_true")
    args = parser.parse_args()
    m = Main()
    m.main(path_documents=args.documents)

#Main()

"""
init
model
load
predict
"""

"""documents = {'head':"Mr. _ and President Obama went on vacations the 1st of September.",
                    'body':"Hello everybody"}

for entity in self.entities:
    if len(entity.documents) > 1:#2,3,4
        print(entity.text, entity.documents)
for idx in [2,3,4]:
    for text in self.documents[idx].entities:
        if 'American' in text.text:
            print(text.text)
#print(self.documents[2].entities.text,self.documents[3].entities.text,self.documents[4].entities.text)
"""