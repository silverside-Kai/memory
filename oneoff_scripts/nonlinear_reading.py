os.chdir('/Users/yihongwang/Projects/mcg/memory')
from langchain_community.document_loaders import GitbookLoader, UnstructuredEPubLoader
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
from neo4j import GraphDatabase

hf_evaluator = load_evaluator("pairwise_embedding_distance",
                              embeddings=embeddings,
                              distance_metric=EmbeddingDistance.COSINE)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=10)


# loader = UnstructuredEPubLoader("./docs/McElreath, Richard_ - Statistical Rethinking_ A Bayesian Course with Examples in R and STAN-CRC Press LLC (2020).epub")
loader = UnstructuredEPubLoader("./docs/McElreath, Richard_ - Statistical Rethinking_ A Bayesian Course with Examples in R and STAN-CRC Press LLC (2020).epub")
data = loader.load()
content = data[0].page_content
current_doc_content = content[27735:87240]
# current_doc_content = content[88032:125568]
# loader = GitbookLoader('https://documents.bevm.io/')
# page_data = loader.load()
# content = page_data[0].page_content

class KeywordContentGraph:
    def __init__(self,
                 max_num_keywords):
        self.docs = []
        self.docs_meta = []
        self.keywords = []
        self.keyword_pagerank_result = []
        self.graph = nx.DiGraph()
        self.max_num_keywords = max_num_keywords
    
    def keyword_rank(self, contents, max_num_keywords):
        llm = LiteLLM("ollama/mistral")
        kw_model = KeyLLM(llm=llm)
        # kw_model = KeyBERT()
        sorted_keywords_scores = []
        for content in contents:
            keywords = kw_model.extract_keywords(content)
            keywords = [[word.lower() for word in sublist] for sublist in keywords]
            print(keywords)
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
    
    def kcg(self):
        current_doc = text_splitter.split_text(current_doc_content)
        self.docs.append(current_doc)
        current_num_pieces = len(current_doc)
        current_doc_meta = [f'{piece_id}' for piece_id in range(current_num_pieces)]
        print(current_doc_meta)
        self.docs_meta.append(current_doc_meta)
        current_keywords = self.keyword_rank(current_doc, max_num_keywords=self.max_num_keywords)
        self.keywords.append(current_keywords)
        self.construct_network()
        current_keyword_pagerank_result = self.pagerank_calc()
        self.keyword_pagerank_result.append(current_keyword_pagerank_result)

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
        # Add nodes and edges to pyvis network
        for node, status in self.graph.nodes(data=True):
            net.add_node(node, shape='box')

        for edge in self.graph.edges(data=True):
            src, dest, data = edge
            net.add_edge(src, dest, label=data['label'])

        # Save the network as an HTML file
        net.write_html(f'./outputs/{output_file_name}.html')

    def pagerank_calc(self):
        undirected_pagerank = nx.pagerank(self.graph)

        sorted_nodes = sorted(undirected_pagerank.items(), key=lambda x: x[1], reverse=True)
        return sorted_nodes


keyword_content_graph_cm = KeywordContentGraph(max_num_keywords=15)
keyword_content_graph_cm.kcg()
keyword_content_graph_cm.network_viz('sr_20240309')


def rag_based_on_keywords(keyword_content_graph, from_node, to_node):
    if nx.has_path(keyword_content_graph.graph, from_node, to_node):
        shortest_path = nx.shortest_path(keyword_content_graph.graph, from_node, to_node)
        print("The shortest path is: ")
        print(shortest_path)
        edge_ids = []
        for i in range(len(shortest_path) - 1):
            if keyword_content_graph.graph.has_edge(shortest_path[i], shortest_path[i+1]):
                edge_ids.append(keyword_content_graph.graph.get_edge_data(shortest_path[i], shortest_path[i+1])['label'])
        if edge_ids:
            link_content = [keyword_content_graph.docs[0][int(link_content_id)] for link_content_id in edge_ids]
            print(f'Original content: {link_content}')
            responses = []
            for idx, content in enumerate(link_content):
                response = simple_prompt(f"""can you summarise the things I need to learn about {shortest_path[idx+1]},
                                        in 1 sentence fewer than 100 words,
                                        given that I already understand what is {shortest_path[idx]},
                                        based on the following text {content}""")
                responses.append(response)
            return responses
        else:
            return None
    else:
        return None


# # Function to upload NetworkX DiGraph to Neo4j
# def upload_to_neo4j(graph, driver):
#     with driver.session() as session:
#         # Iterate over nodes and create them in Neo4j
#         for node_id in graph.nodes:
#             session.run("CREATE (:Node {id: $id})", id=node_id)

#         # Iterate over edges and create relationships in Neo4j
#         for source, target in graph.edges:
#             session.run("MATCH (source:Node {id: $source}), (target:Node {id: $target}) "
#                         "CREATE (source)-[:CONNECTED_TO]->(target)", 
#                         source=source, target=target)
            
# upload_to_neo4j(keyword_content_graph.graph, driver)

# graph = nx.DiGraph()
# graph.add_nodes_from([1, 2, 3])
# graph.add_edges_from([(1, 2), (2, 3)])

# # Upload the graph to Neo4j
# upload_to_neo4j(graph, driver)