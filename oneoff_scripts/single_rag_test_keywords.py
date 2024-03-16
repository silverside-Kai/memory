from langchain_community.document_loaders import GitbookLoader
from langchain.tools import DuckDuckGoSearchRun
from modules.prompts.simple_prompt_litellm import simple_prompt
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from keybert.llm import LiteLLM
from keybert import KeyLLM, KeyBERT
from modules.load_data.load_vector_stores import embeddings
from langchain.evaluation import load_evaluator, EmbeddingDistance
from modules.misc.max_length import truncate_string
import networkx as nx
import os
os.chdir('/Users/yihongwang/Projects/mcg/memory')
hf_evaluator = load_evaluator("pairwise_embedding_distance",
                              embeddings=embeddings,
                              distance_metric=EmbeddingDistance.COSINE)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)


# loader = GitbookLoader('https://documents.bevm.io/')
# page_data = loader.load()
# content = page_data[0].page_content

question_0 = 'Who is Mitoma?'

class KeywordContentGraph:
    def __init__(self,
                 iter_num,
                 question_0,
                 max_num_keywords,
                 num_keywords_for_questions,
                 search_methods):
        self.iter_num = iter_num
        self.questions = []
        self.docs = []
        self.docs_meta = []
        self.keywords = []
        self.keyword_pagerank_result = []
        self.search_methods = search_methods
        self.graph = nx.DiGraph()
        self.max_num_keywords = max_num_keywords
        self.num_keywords_for_questions = num_keywords_for_questions
        self.top_n_keywords_for_questions = []
        self.questions.append(question_0)
    
    def search(self, query, search_method):
        if search_method=='ddg':
            content = DuckDuckGoSearchRun().run(query)
        elif search_method=='wiki':
            content = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()).run(query)
        else:
            content = simple_prompt(query)
        return content
    
    def keyword_rank(self, contents, max_num_keywords):
        llm = LiteLLM("ollama/mistral")
        kw_model = KeyLLM(llm=llm)
        # kw_model = KeyBERT()
        sorted_keywords_scores = []
        for content in contents:
            keywords = kw_model.extract_keywords(content)
            # keywords = [[tup[0] for tup in keywords]]
            keyword_scores = []
            for i in range(len(keywords)):
                for j in range(len(keywords[i])):
                    keyword_ij = keywords[i][j]
                    distance_score = hf_evaluator.evaluate_string_pairs(
                        prediction=content, prediction_b=keyword_ij)
                    keyword_score = {keyword_ij: distance_score['score']}
                    keyword_scores.append(keyword_score)
            sorted_keyword_scores = sorted(keyword_scores, key=lambda x: list(x.values())[0], reverse=True)
            sorted_keywords_scores.append(sorted_keyword_scores[-max_num_keywords:])
        return sorted_keywords_scores
    
    def recursive_kcg(self):

        for i in range(self.iter_num):
            search_methods = self.search_methods
            for search_method_id, search_method in enumerate(search_methods):
                current_doc_index = len(search_methods)*i+search_method_id
                current_doc_content = self.search(self.questions[i], search_method)
                current_piece_content = text_splitter.split_text(current_doc_content)
                self.docs.append(current_piece_content)
                current_num_pieces = len(current_piece_content)
                current_doc_meta = [f'{i}_{search_method}_{piece_id}' for piece_id in range(current_num_pieces)]
                print(current_doc_meta)
                self.docs_meta.append(current_doc_meta)
                current_doc = self.docs[current_doc_index]
                current_keywords = self.keyword_rank(current_doc, max_num_keywords=self.max_num_keywords)
                self.keywords.append(current_keywords)
            self.construct_network()
            current_keyword_pagerank_result = self.pagerank_calc()
            self.keyword_pagerank_result.append(current_keyword_pagerank_result)
            existing_keywords_for_questions = set([item for sublist in self.top_n_keywords_for_questions for item in sublist])
            current_keyword_pagerank_result_filtered = [tup[0] for tup in current_keyword_pagerank_result if tup[0] not in existing_keywords_for_questions]
            current_top_n_keywords_for_questions = current_keyword_pagerank_result_filtered[:self.num_keywords_for_questions]
            print(current_top_n_keywords_for_questions)
            self.top_n_keywords_for_questions.append(current_top_n_keywords_for_questions)
            next_question = simple_prompt(f'Given the top keywords {current_top_n_keywords_for_questions}, ask a relevant question in 50 words.')
            next_question = truncate_string(next_question, max_length=250)
            print(f'next question: {next_question}')
            if len(next_question) == 0:
                next_question = str(current_top_n_keywords_for_questions)
            self.questions.append(next_question)

    def harmonise_keywords(self):
        for keywords_per_doc in self.keywords:
            continue
    
    def construct_network(self):
        # Create a directed graph
        for doc_id, keywords_per_doc in enumerate(self.keywords):
            for piece_id, keywords_per_piece in enumerate(keywords_per_doc):
                piece_meta = self.docs_meta[doc_id][piece_id]
                for kw_dict in keywords_per_piece:
                    kw = list(kw_dict.keys())[0]
                    if kw not in self.graph:
                        self.graph.add_node(kw, status=int(piece_meta.split('_')[0]))
                for kw_dict in keywords_per_piece:
                    kw = list(kw_dict.keys())[0]
                    for kw_2_dict in keywords_per_piece:
                        kw_2 = list(kw_2_dict.keys())[0]
                        if kw != kw_2:
                            self.graph.add_edge(kw, kw_2, label=piece_meta)

    def network_viz(self, output_file_name):
        from pyvis.network import Network
        # Create a pyvis network
        net = Network(notebook=False, width='100%', height='800px')
        status_color_map = {0: 'pink', 1: 'green', 2: 'yellow', 3: 'cyan'}
        # Add nodes and edges to pyvis network
        for node, status in self.graph.nodes(data=True):
            node_color = status_color_map.get(status.get('status', 1), 'grey')
            net.add_node(node, shape='box', color=node_color)

        for edge in self.graph.edges(data=True):
            src, dest, data = edge
            net.add_edge(src, dest, label=data['label'])

        # Save the network as an HTML file
        net.write_html(f'./outputs/{output_file_name}.html')

    def pagerank_calc(self):
        undirected_pagerank = nx.pagerank(self.graph)

        sorted_nodes = sorted(undirected_pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes


keyword_content_graph = KeywordContentGraph(iter_num=3,
                                            question_0=question_0,
                                            max_num_keywords=5,
                                            num_keywords_for_questions=5,
                                            search_methods=['ddg','wiki','llm'])
keyword_content_graph.recursive_kcg()
keyword_content_graph.network_viz('causal_dag_20240208_fm')

graph_data = nx.node_link_data(keyword_content_graph.graph)

link_count = {}
# Iterate through the edges and count the links
for edge in graph_data['links']:
    source = edge['source']
    target = edge['target']
    
    # Create a tuple representing the edge
    edge_tuple = (source, target)
    
    # Update the count for the edge in the dictionary
    link_count[edge_tuple] = link_count.get(edge_tuple, 0) + 1

# Print the link_count dictionary to check if it contains the expected data
print("Link Count Dictionary:", link_count)

# Extract cases where there are more than 1 link
multiple_links = [(edge, count) for edge, count in link_count.items() if count > 1]

# Print the multiple_links list to check if it contains any items
print("Multiple Links:", multiple_links)


def network_viz(graph, output_file_name):
    from pyvis.network import Network
    # Create a pyvis network
    net = Network(notebook=False, width='100%', height='800px')
    status_color_map = {0: 'pink', 1: 'green', 2: 'yellow'}
    # Add nodes and edges to pyvis network
    for node, status in graph.nodes(data=True):
        node_color = status_color_map.get(status.get('status', 1), 'grey')
        net.add_node(node, shape='box', color=node_color)

    for edge in graph.edges(data=True):
        src, dest, data = edge
        net.add_edge(src, dest, label=data['label'])

    # Save the network as an HTML file
    net.write_html(f'./outputs/{output_file_name}.html')

def construct_network(keywords, docs_meta):
    # Create a directed graph
    graph = nx.DiGraph()
    for doc_id, keywords_per_doc in enumerate(keywords):
        for piece_id, keywords_per_piece in enumerate(keywords_per_doc):
            piece_meta = docs_meta[doc_id][piece_id]
            for kw_dict in keywords_per_piece:
                kw = list(kw_dict.keys())[0]
                if kw not in graph:
                    graph.add_node(kw, status=int(piece_meta.split('_')[0]))
            for kw_dict in keywords_per_piece:
                kw = list(kw_dict.keys())[0]
                for kw_2_dict in keywords_per_piece:
                    kw_2 = list(kw_2_dict.keys())[0]
                    if kw != kw_2:
                        graph.add_edge(kw, kw_2, label=piece_meta)
    return graph