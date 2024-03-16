
from langchain_community.document_loaders import GitbookLoader, WebBaseLoader
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatLiteLLM
from langchain.memory import ConversationKGMemory
from langchain.tools import DuckDuckGoSearchRun
from modules.prompts.simple_prompt_litellm import simple_prompt
from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from keybert.llm import LiteLLM
from keybert import KeyLLM
from modules.load_data.load_vector_stores import embeddings
from langchain.evaluation import load_evaluator, EmbeddingDistance


# loader = GitbookLoader('https://documents.bevm.io/')
# page_data = loader.load()
# content = page_data[0].page_content

content = simple_prompt('What is PCI compliance? Why it is important to online payment?')

llm = LiteLLM("ollama/mistral")
kw_model = KeyLLM(llm=llm)
keywords = kw_model.extract_keywords(content)


def rank_keywords(keywords):
    hf_evaluator = load_evaluator("pairwise_embedding_distance",
                                  embeddings=embeddings,
                                  distance_metric=EmbeddingDistance.COSINE)
    keyword_scores = []
    for i in range(len(keywords)):
        for j in range(len(keywords[i])):
            keyword_ij = keywords[i][j]
            print(keyword_ij)
            distance_score = hf_evaluator.evaluate_string_pairs(
                prediction=content, prediction_b=keyword_ij)
            keyword_score = {keyword_ij: distance_score['score']}
            keyword_scores.append(keyword_score)
    return keyword_scores

keyword_scores = rank_keywords(keywords)
sorted_keyword_scores = sorted(keyword_scores, key=lambda x: list(x.values())[0], reverse=True)

chat = ChatLiteLLM(model="ollama/mistral")

memory = ConversationKGMemory(llm=chat)
kgs = memory.get_knowledge_triplets(content)

ddg_contents = []
wiki_contents = []
for i in range(3):
    next_keyword = list(sorted_keyword_scores[-(i+1)].keys())[0]

    search = DuckDuckGoSearchRun()
    ddg_content = search.run(f"What is {next_keyword}?")
    extended_kgs_ddg = memory.get_knowledge_triplets(ddg_content)
    kgs.extend(extended_kgs_ddg)
    # ddg_contents.append(ddg_content)
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    wiki_content = wikipedia.run(f"{next_keyword}")
    extended_kgs_wiki = memory.get_knowledge_triplets(wiki_content)
    # wiki_contents.append(wiki_content)
    kgs.extend(extended_kgs_wiki)


def network_viz(kgs, output_file_name):
    import networkx as nx
    from pyvis.network import Network
    # Create a directed graph
    graph = nx.DiGraph()

    # Add nodes and edges to the graph
    for kg in kgs:
        if kg.subject not in graph:
            graph.add_node(kg.subject)
        if kg.object_ not in graph:
            graph.add_node(kg.object_)
        graph.add_edge(kg.subject, kg.object_, label=kg.predicate)

    # Create a pyvis network
    net = Network(notebook=False, width='100%', height='800px')

    # Add nodes and edges to pyvis network
    for node in graph.nodes():
        net.add_node(node, shape='box')

    for edge in graph.edges(data=True):
        src, dest, data = edge
        net.add_edge(src, dest, label=data['label'])

    # Save the network as an HTML file
    net.write_html(f'./outputs/{output_file_name}.html')

network_viz(kgs, 'knowledge_graph_pci_svo')