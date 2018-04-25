import pandas as pd
import numpy as np

from py2neo import authenticate, Graph, Node, Relationship, NodeSelector

import subprocess
import webbrowser
import sys
import itertools

# make it with only merge not create unique
# MERGE (n { name:"X" })-[r:RELATED]->(m) DELETE r, n, m;
class GraphObject:

    def __init__(self, entities={}, ent2idx={}, documents=[]):
        self.entities = entities
        self.ent2idx = ent2idx
        self.documents = documents
        self.nodes = []
        self.user = 'neo4j'
        self.password = "f1234567"

        self.launchneo4j(popup=False)
        ####################### self.graph.delete_all()

    def populate(self, entities, ent2idx, documents, queries):
        self.entities = entities
        self.ent2idx = ent2idx
        self.documents = documents
        self.queries = queries

    def add_ent_to_doc(self, document, type='DOCUMENT'):
        n_document = {'properties': {"name": ("Doc " + str(document.id)), "title": document.head}, 'type': type,
                  'node_name': "doc" }
        count = 0
        n_entities = []
        entities = []
        for entity in document.entities:
            node_name = ''.join(['`', str(self.entities[entity].text), self.entities[entity].label, '`'])
            if node_name not in entities:
                entities.append(node_name)
                n_entities.append({'properties': {"name": str(self.entities[entity].text)}, 'type': self.entities[entity].label, 'node_name': node_name})
                count += 1
            # I do it by batches because it blocks otherwise
            if count>5:
                self.match_relationship(n_document, n_entities, 'APPEARS_IN')
                count=0
                n_entities = []
        if count >= 1:
            self.match_relationship(n_document, n_entities, 'APPEARS_IN')

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ do match at the beggining to identify the document, then refer to it
    def match_relationship(self, n_document, n_entities, relationship):
        # Build the queries in form of strings to be able to add all the entities at once to a document.
        CREATE_entities_string = ['''MERGE ('''+n_document['node_name']+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+'''", title: "'''+n_document['properties']['title']+'''"})\n''']
        for num, node in enumerate(n_entities):
            #MERGE_str = '''MERGE ('''+node['node_name']+''':'''+node['type']+''' {name: "'''+node['properties']['name']+'''"})'''
            ##############################CREATE_str = '''MERGE ('''+n_document['node_name']+str(num)+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+'''", title: "'''+n_document['properties']['title']+'''"})'''+'''<-[:'''+relationship+''']-('''+node['node_name']+''':'''+node['type']+''' {name: "'''+node['properties']['name']+'''"})'''
            CREATE_str = '''MERGE ('''+n_document['node_name']+''')''' + '''<-[:''' + relationship + ''']-(''' + node['node_name'] + ''':''' + node['type'] + ''' {name: "''' + node['properties']['name'] + '''"})'''
            #MERGE_entities_string.append(MERGE_str)

            if num == len(n_entities)-1:
                #CREATE_str += ''';'''
                CREATE_str += ''''''

            else:
                ##########CREATE_str += '''\nMERGE '''
                CREATE_str += '''\n'''
            CREATE_entities_string.append(CREATE_str)


        #MERGE_entities_string = '''\n'''.join(MERGE_entities_string)
        CREATE_entities_string = ''''''.join(CREATE_entities_string)        #print(CREATE_entities_string)

        query = CREATE_entities_string
        #print(query)
        self.graph.evaluate(query)

    def match_relationship_GOOD(self, n_document, n_entities, relationship):
        # Build the queries in form of strings to be able to add all the entities at once to a document.
        MERGE_entities_string = ['''MERGE ('''+n_document['node_name']+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+'''", title: "'''+n_document['properties']['title']+'''"})''']
        CREATE_entities_string =[]
        print("aaaaaaaaaaaaaaaaaaaaaaa")
        for num, node in enumerate(n_entities):
            MERGE_str = '''MERGE ('''+node['node_name']+''':'''+node['type']+''' {name: "'''+node['properties']['name']+'''"})'''
            CREATE_str = '''('''+n_document['node_name']+''')<-[:'''+relationship+''']-('''+node['node_name']+''')'''
            MERGE_entities_string.append(MERGE_str)

            if num == len(n_entities)-1:
                CREATE_str += ''';'''
                ################3CREATE_str += ''''''

            else:
                ##########CREATE_str += '''\nMERGE '''
                CREATE_str += ''','''
            CREATE_entities_string.append(CREATE_str)


        MERGE_entities_string = '''\n'''.join(MERGE_entities_string)
        CREATE_entities_string = '''\n'''.join(['CREATE UNIQUE ', '''\n'''.join(CREATE_entities_string)])        #print(CREATE_entities_string)
        ###################3#CREATE_entities_string = '''\n'''.join(['MERGE ', '''\n'''.join(CREATE_entities_string)])        #print(CREATE_entities_string)

        query = ''''''.join([MERGE_entities_string,CREATE_entities_string])
        print(query)
        self.graph.evaluate(query)

    def add_similarity_to_doc(self, document, type='DOCUMENT'):
        n_document = {'properties': {"name": ("Doc " + str(document.id)), "title": document.head}, 'type': type,
                  'node_name': "doc" }#+ str(document.id) # ''.join(['`',document.head,'`'])
        count = 0
        relationship = []
        n_documents = []
        for key, similar_document in document.similarities.items():
            prob = document.similarities[key]['prob']
            idx = document.similarities[key]['idx']
            if idx < len(self.documents) and idx != document.id:
                n_documents.append({'properties': {"name": "Doc " + str(self.documents[idx].id)}, 'type': 'DOCUMENT', 'node_name': ''.join(['`',str(self.documents[idx].head),'`'])})
                relationship.append("""SIMILAR_TO{prob:"""+str(prob)+"""}""")
                count +=1
        if count >= 1:
            self.match_similarity_relationship(n_document, n_documents, relationship)

    def add_similarity_to_query(self, query, type='QUERY'):
        n_document = {'properties': {"name": ("Doc " + str(query.id)), "title": query.head}, 'type': type,
                  'node_name': "doc" }#+ str(document.id) # ''.join(['`',document.head,'`'])
        count = 0
        relationship = []
        n_documents = []
        for key, similar_document in query.similarities.items():
            prob = query.similarities[key]['prob']
            idx = query.similarities[key]['idx']
            if idx < len(self.documents):
                n_documents.append({'properties': {"name": "Doc " + str(self.documents[idx].id)}, 'type': 'DOCUMENT', 'node_name': ''.join(['`',str(self.queries[idx].head),'`'])})
                relationship.append("""SIMILAR_TO{prob:"""+str(prob)+"""}""")
                count +=1
        if count >= 1:
            self.match_similarity_relationship(n_document, n_documents, relationship)

    def match_similarity_relationship(self, n_document, n_entities, relationship):
        # Build the queries in form of strings to be able to add all the entities at once to a document.
        MERGE_entities_string = ['''MERGE ('''+n_document['node_name']+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+'''", title: "'''+n_document['properties']['title']+'''"})''']
        CREATE_entities_string =[]
        for num, node in enumerate(n_entities):
            MERGE_str = '''MERGE ('''+node['node_name']+''':'''+node['type']+''' {name: "'''+node['properties']['name']+'''"})'''
            CREATE_str = '''('''+n_document['node_name']+''')<-[:'''+relationship[num]+''']-('''+node['node_name']+''')'''
            MERGE_entities_string.append(MERGE_str)

            if num == len(n_entities)-1:
                CREATE_str += ''';'''
            else:
                CREATE_str += ''','''
            CREATE_entities_string.append(CREATE_str)

        MERGE_entities_string = '''\n'''.join(MERGE_entities_string)
        CREATE_entities_string = '''\n'''.join(['CREATE UNIQUE ','''\n'''.join(CREATE_entities_string)])
        #print(CREATE_entities_string)

        query = ''''''.join([MERGE_entities_string,CREATE_entities_string])
        self.graph.evaluate(query)

    # Entities should be an array of entity objects
    def find_documents_on_entities(self, entities):
        """MATCH (n)
        WHERE n.number = 1
        OPTIONAL MATCH n-[r*..2]-(c)
        RETURN c as nodes, collect(r) as sublinks"""
        n_entities = []
        for entity in entities:
            n_entities.append({'properties': {"name": str(self.entities[entity].text)}, 'type': self.entities[entity].label, 'node_name': ''.join(['`',str(self.entities[entity].text),self.entities[entity].label,'`'])})

        WHERE_entities_string = []
        for num, node in enumerate(n_entities):
            WHERE_str = '''n.name = "'''+node['properties']['name']+'''"'''

            if num != len(n_entities) - 1:
                WHERE_str += ''' or '''

            WHERE_entities_string.append(WHERE_str)
        # MATCH (x:User), (y:User) we could do it like this as well
        WHERE_entities_string = '''\n'''.join(['MATCH (n)\nWHERE ', '''\n'''.join(WHERE_entities_string)])
        WHERE_entities_string += '''\nOPTIONAL MATCH (n)-[r*..2]-(c)\n'''
        WHERE_entities_string += '''\nRETURN collect(c) as nodes'''

        #WHERE_entities_string += '''\nRETURN collect(n) as nodes, collect(r) as sublinks'''

        query = WHERE_entities_string
        #print(query)
        response = self.graph.evaluate(query)

        return response

    def launchneo4j(self, popup = True):
        # connect to authenticated graph database
        authenticate("localhost:7474", self.user, self.password)
        self.graph = Graph("http://localhost:7474/db/data/")
        if popup:
            url = "http://localhost:7474/browser/"
            if sys.platform == 'darwin':  # in case of OS X
                subprocess.Popen(['open', url])
            else:
                webbrowser.open_new_tab(url)


