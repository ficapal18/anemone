import pandas as pd
import numpy as np

from py2neo import authenticate, Graph, Node, Relationship, NodeSelector

import subprocess
import webbrowser
import sys
import itertools


class GraphObject:

    def __init__(self, entities, ent2idx):
        self.entities = entities
        self.ent2idx = ent2idx
        self.nodes = []
        self.user = 'neo4j'
        self.password = "f1234567"

        self.launchneo4j(popup=False)
        ################################### self.graph.delete_all()

    def add_ent_to_doc(self, document):
        n_document = {'properties': {"name": ("Doc " + str(document.id)), "title": document.head}, 'type': 'DOCUMENT',
                  'node_name': "doc" }#+ str(document.id) # ''.join(['`',document.head,'`'])
        n_entities = []
        for entity in document.entities:
            n_entities.append({'properties': {"name": str(self.entities[entity].text)}, 'type': self.entities[entity].label, 'node_name': ''.join(['`',str(self.entities[entity].text),self.entities[entity].label,'`'])})

        self.match_relationship(n_document, n_entities, 'APPEARS_IN')

    def match_relationship(self, n_document, n_entities, relationship):
        # Build the queries in form of strings to be able to add all the entities at once to a document.
        MERGE_entities_string = ['''MERGE ('''+n_document['node_name']+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+'''", title: "'''+n_document['properties']['title']+'''"})''']
        CREATE_entities_string =[]
        for num, node in enumerate(n_entities):
            MERGE_str = '''MERGE ('''+node['node_name']+''':'''+node['type']+''' {name: "'''+node['properties']['name']+'''"})'''
            CREATE_str = '''('''+n_document['node_name']+''')<-[:'''+relationship+''']-('''+node['node_name']+''')'''
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
        #print(query)
        """
        query ='''
           MERGE ('''+n_document['node_name']+''':'''+n_document['type']+''' {name: "'''+n_document['properties']['name']+''','''+n_document['properties']['title']+'''"})
           MERGE ('''+n2['node_name']+''':'''+n2['type']+''' {name: "'''+n2['properties']['name']+'''"})
           CREATE UNIQUE ('''+n1['node_name']+''')-[:'''+relationship+''']->('''+n2['node_name']+''')
        '''
        """
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
        print(query)
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


