import time
import src.document

class Query:

    def __init__(self, id, body=""):
        self.ts = time.time()
        self.id = id
        self.body = body
        self.head = body
        self.entities = []
        self.similarities = {}
        self.similarities_to_queries = {}

    def add_entities(self, entities):
        self.entities.extend(entities)

    def add_similarities(self, similarities):
        self.similarities = similarities