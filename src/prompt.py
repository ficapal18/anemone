import cmd, sys


class Prompt(cmd.Cmd):
    intro = '\nWelcome to Anemone search engine. Type help or ? to obtain a list of commands.\n'

    def __init__(self, main_instance, stufflist=[]):
        self.main_instance = main_instance
        cmd.Cmd.__init__(self)
        self.prompt = '>>> '
        self.stufflist = stufflist

    def do_quit(self, arg):
        sys.exit(0)

    def do_print_stuff(self, arg):
        for s in self.stufflist:
            print(s)

    def do_search(self, arg):
        'Perform a search query'
        search = input("Please enter something: ")
        self.main_instance.similarity_object.query_to_lda(search, self.main_instance.document_files)
        #self.main_instance.similarity_object.query_to_tf_idf(search, self.main_instance.document_files)