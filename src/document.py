

class Document:

    def __init__(self, id, head="", body=""):
        self.id = id
        self.head = head
        self.body = body
        self.entities = []

    def add_entities(self, entities):
        self.entities.extend(entities)