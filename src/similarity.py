import gensim
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import string
import pickle
from time import time

class Similarity:

    def __init__(self, path='./data/similarity/'):

        self.path = path
        self.gen_docs = None
        self.dictionary = None
        self.corpus = None

        # TF-IDF Variables
        self.tf_idf = None
        self.tf_idf_index = None

        # LDA Variables
        self.lda = None
        self.lda_index = None


    # Tokenizes an array of strings and gives an array of arrays of strings as output
    def tokenize_documents(self, documents, language='english'):
        punct_array = string.punctuation + "”’•"
        translator = str.maketrans('', '', punct_array)
        gen_docs = [
            [w.lower() for w in word_tokenize(text.translate(translator)) if w.lower() not in stopwords.words(language)]
            for text in documents]
        #gen_docs = [[w.lower() for w in word_tokenize(text) if w.lower() not in stopwords.words(language)] for text in documents]
        with open(self.path+'gen_docs', 'wb') as handle:
            pickle.dump(gen_docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.gen_docs = gen_docs
        print("The documents have been tokenized and the gen doc is available.")
        return gen_docs

    # Generates the corpus
    def generate_corpus(self, gen_docs):
        dictionary = gensim.corpora.Dictionary(gen_docs)
        corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
        dictionary.save(self.path+'/dictionary.dict')
        gensim.corpora.MmCorpus.serialize(self.path+'/corpus.mm', corpus)
        self.corpus = corpus
        self.dictionary = dictionary
        print("The corpus is available.")
        return corpus, dictionary

    # Compute tf-idf from the input corpus, create similarity matrix of all files
    def tf_idf_train(self):
        print("Computing the TF-IDF Model.")
        t0 = time()
        #corpus = gensim.corpora.MmCorpus(self.path+'/corpus.mm')
        # Transform Text with TF-IDF, we convert our vectors corpus to TF-IDF space
        tf_idf = gensim.models.TfidfModel(self.corpus)
        tf_idf.save(self.path+'tf_idf.model')
        index = gensim.similarities.SparseMatrixSimilarity(tf_idf[self.corpus], num_features=12)
        index.save(self.path+'tf_idf.index')
        print("The TF-IDF Model is available, it took:",time()-t0,"s.")
        self.tf_idf_index = index
        self.tf_idf = tf_idf
        return tf_idf

    # Transforms a string into a format that we can use to query our tdfidf similary matrix
    def query_to_tdfidf(self, string):
        # In order to query the dictionary and corpus must have been loaded in memory
        assert self.dictionary is not None
        assert self.corpus is not None

        query_doc = [w.lower() for w in word_tokenize(string)]
        query_doc_bow = self.dictionary.doc2bow(query_doc)
        query_doc_tf_idf = self.tf_idf[query_doc_bow]
        return query_doc_tf_idf

    # Compute lda from the input corpus, create similarity matrix of all files
    def lda_train(self):
        print("Computing the LDA Model.")
        t0 = time()
        corpus = gensim.corpora.MmCorpus(self.path+'/corpus.mm')
        # Transform Text with TF-IDF
        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics=200)
        lda.save(self.path+'lda.model')
        print("The LDA model has been computed!")
        # We convert our vectors corpus to TF-IDF space
        index = gensim.similarities.MatrixSimilarity(lda[corpus])
        index.save(self.path+'lda_similarity.index')
        print("The LDA Model is available, it took:",time()-t0,"s.")
        self.lda_index = index
        self.lda = lda
        return lda

    def query_to_lda(self, doc, document_files=[], language='english', k=20, verbose = True):
        punct_array = string.punctuation + "”’•"
        translator = str.maketrans('', '', punct_array)
        gen_docs = [w.lower() for w in word_tokenize(doc.translate(translator)) if w.lower() not in stopwords.words(language)]
        vec_bow = self.dictionary.doc2bow(gen_docs)#doc.lower().split())
        vec_lsi = self.lda[vec_bow]  # convert the query to LSI space

        top_k = {}
        sims = self.lda_index[vec_lsi]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        if verbose:
            print("\nSearching Query:", doc)
            print("\nSimilar Documents:")
        for i in range(0,k):
            idx, prob = sims[i]
            if verbose:
                print(i+1,str('{0:.2f}'.format(prob)),"\t", document_files.iloc[idx]["head"])
            top_k[i]={}
            top_k[i]['idx'] = idx
            top_k[i]['prob'] = prob
        return top_k

    def query_to_tf_idf(self, doc, document_files):
        print("\nSearching Query:", doc)

        vec_bow = self.dictionary.doc2bow(doc.lower().split())
        vec_tf_idf = self.tf_idf[vec_bow]  # convert the query to LSI space

        sims = self.tf_idf_index[vec_tf_idf]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        print("\nSimilar Documents:")
        for i in range(0,10):
            idx, prob = sims[i]
            print(i+1,"\t", document_files.iloc[idx]["head"])

    def load_tf_idf(self):
        tf_idf = gensim.models.ldamodel.LdaModel.load(self.path+'tf_idf.model')
        self.tf_idf = tf_idf
        return tf_idf

    def load_tf_idf_index(self):
        tf_idf_index = gensim.similarities.SparseMatrixSimilarity.load(self.path+'tf_idf.index')
        self.lda_index = tf_idf_index
        return tf_idf_index

    def load_lda(self):
        lda = gensim.models.ldamodel.LdaModel.load(self.path+'lda.model')
        self.lda = lda
        return lda

    def load_lda_index(self):
        lda_index = gensim.similarities.MatrixSimilarity.load(self.path+'lda_similarity.index')
        self.lda_index = lda_index
        return lda_index

    # Loads corpus in memory
    def load_corpus(self):
        self.corpus = gensim.corpora.MmCorpus(self.path+'/corpus.mm')
        return self.corpus

    # Loads the dictionary in memory
    def load_dictionary(self):
        self.dictionary = gensim.corpora.Dictionary.load(self.path+'/dictionary.dict')
        return self.dictionary

    # Loads the gen_docs in memory
    def load_gendocs(self):
        with open(self.path+'gen_docs', 'rb') as handle:
            self.gen_docs = pickle.load(handle)

