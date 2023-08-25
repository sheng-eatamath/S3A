import networkx as nx
from networkx.algorithms.traversal.breadth_first_search import bfs_predecessors, bfs_successors
from networkx.algorithms.traversal.depth_first_search import dfs_predecessors, dfs_successors
from copy import deepcopy
from networkx.drawing.nx_pydot import write_dot
from functools import reduce

def create_graph(nouns, vocab_ids, vocab_names, vocab_def):
    G = nx.DiGraph()
    for i, n in enumerate(nouns):
        G.add_node(n.name(),
                   synid=n.name(),
                   id=vocab_ids[i],
                   classname=vocab_names[i],
                   definition=vocab_def[i].replace(':', ' - '),
                   depth=n.max_depth(),
                  )

    queue = nouns
    while len(queue):
        n = queue.pop(0)
        children = n.hyponyms()
        parents = n.hypernyms()
        for ch in children:
            if not G.has_edge(n.name(), ch.name()):
                G.add_edge(n.name(), ch.name())
        for pr in parents:
            if not G.has_edge(pr.name(), n.name()):
                G.add_edge(pr.name(), n.name())
        queue.extend(children)
    return G


def predecessor_set(G, source, depth=-1):
    if depth==-1:
        queue = list(G.predecessors(source)) + [source]
        result_set = set(queue)
        while len(queue):
            item = queue.pop(0)
            # if item in G.nodes:
            pre = list(G.predecessors(item))
            queue.extend(pre)
            result_set |= set(pre)
    else:
        queue = [[source]]
        for i in range(depth):
            add_queue = set()
            for item in queue[-1]:
                add_queue |= set(list(G.successors(item)))
            queue.append(list(add_queue))
        result_set = set(reduce(lambda x, y: x+y, queue))
    return result_set

def successor_set(G, source, depth=3):
    queue = [[source]]
    for i in range(depth):
        add_queue = set()
        for item in queue[-1]:
            add_queue |= set(list(G.successors(item)))
        queue.append(list(add_queue))
    result_set = set(reduce(lambda x, y: x+y, queue))
    return result_set

def predecessor_list(G, source, depth=3):
    queue = [[source]]
    for i in range(depth):
        add_queue = set()
        for item in queue[-1]:
            add_queue |= set(list(G.successors(item)))
        queue.append(list(add_queue))
    result_set = reduce(lambda x, y: x+y, queue)
    return result_set
