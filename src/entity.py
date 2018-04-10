
class Entity:

    def __init__(self, id, text, label):
        self.id = id
        self.text = text
        self.label = label
        self.documents = []
        self.alias = []

    def append_document(self, document):
        self.documents.append(document)

    def append_alias(self, alias):
        self.alias.append(alias)