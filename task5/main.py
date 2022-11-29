import doctest
from typing import List, Dict, Any
from copy import deepcopy
from itertools import chain, combinations
from heapq import heapify, heappush, heappop


def bounded_subsets(numbers: list, c):
    """
    return all subsets of numbers such thst there values are less or equal to c
    :param numbers: a list of numbers to generate subsets from
    :param c: upper limit on subsets sums
    :return:

    >>> [sorted(s) for s in bounded_subsets([1, 2, 3], 4)]
    [[], [1], [2], [1, 2], [3], [1, 3]]

    >>> [s for s in zip(range(5), bounded_subsets([i for i in range(100)], 1000000000000))]
    [(0, []), (1, [0]), (2, [1]), (3, [0, 1]), (4, [2])]

    >>> [s for s in bounded_subsets([45, 56, 123, 90], 7)]
    [[]]
    """

    # get the smallest ones first
    numbers.sort()

    # represent a subset as a binary string
    for i in range(2 ** len(numbers)):

        binary_representation = format(i, f'0{len(numbers)}b')

        subset = [num for i, num in enumerate(numbers) if binary_representation[len(numbers) - i - 1] == '1']

        if sum(subset) <= c:
            yield subset


def bounded_subsets_ordered(numbers: list, c):
    """
     return all subsets of numbers such thst there values are less or equal to c, in sorted order of sums
    :param numbers: a list of numbers to generate subsets from
    :param c: upper limit on subsets sums
    :return:

    >>> [s for s in bounded_subsets_ordered([5, 2, 4], 7)]
    [[], [2], [4], [5], [2, 4], [2, 5]]

    >>> [s for s in bounded_subsets([1, 2, 3], 4)]
    [[], [1], [2], [1, 2], [3], [1, 3]]

    >>> [s for s in zip(range(5), bounded_subsets([i for i in range(100)], 1000000000000))]
    [(0, []), (1, [0]), (2, [1]), (3, [0, 1]), (4, [2])]

    >>> [s for s in bounded_subsets([45, 56, 123, 90], 7)]
    [[]]

    """
    yield []

    numbers.sort()

    # no possible non-empty subsets
    if numbers[0] > c:
        return

    # keep all possible next-smallest subsets in a min heap
    # first element is the sum (to be sorted by), second is the subset,
    # and third is index of the next item to be considered
    possible_subsets = [(numbers[0], [numbers[0]], 1)]
    heapify(possible_subsets)

    while possible_subsets:
        current_subset_sum, current_subset, index = heappop(possible_subsets)

        # add next possible subsets to the heap
        if not index == len(numbers) and current_subset_sum < c:
            heappush(possible_subsets,
                     (current_subset_sum + numbers[index], current_subset + [numbers[index]], index + 1))
            heappush(possible_subsets, (current_subset_sum + numbers[index] - current_subset[-1],
                                        current_subset[:-1] + [numbers[index]], index + 1))

        if current_subset_sum <= c:
            yield current_subset


def convert_matrix_to_list_graph(graph: List[List[bool]]) -> Dict[Any, List]:
    """
    convert a ajucency natrix into nieghors list (Dict[Any, list])
    :param graph:
    :return:

    >>> convert_matrix_to_list_graph(\
    [[False, False, True, True], [False, False, False, False], [True, False, False, True], [True, False, True ,False]])
    {0: [2, 3], 1: [], 2: [0, 3], 3: [0, 2]}
    """
    return {i: [j for j in range(len(graph[0])) if graph[i][j]] for i in range(len(graph))}


def greedy(graph: Dict[Any, list]) -> list:
    """
    Greedy search to find minimum vc - take the node that coover the most nodes at each search
    :param graph:
    :return:
    """
    working_graph = deepcopy(graph)

    vc = []
    # while there are still nodes in the graph
    while working_graph:
        max_node = max(working_graph.keys(), key=lambda node: len(working_graph.get(node)))
        vc.append(max_node)

        for nei in working_graph.get(max_node):
            working_graph.pop(nei)
        working_graph.pop(max_node)

    return vc


def full_search(graph: Dict[Any, list]) -> list:
    """
    Full search of all possible subsets of nodes in the graph, to find the minimum vc
    :param graph:
    :return:
    """
    min_vc = list(graph.keys())

    for subset in chain.from_iterable(combinations(graph.keys(), r) for r in range(len(graph) + 1)):
        if len(subset) < len(min_vc) and \
                all([node in subset or any([nei in subset for nei in neighbors]) for node, neighbors in graph.items()]):
            min_vc = subset

    return min_vc


def vertex_cover(algo: callable, graph, output_type: callable = lambda x: x, graph_convertor: callable = lambda x: x,
                 k: int = -1, **kwargs):
    """
    Find a vertex cover in a given garph
    :param algo: Algorithm to find vertex cover. must take graph as a Dict[Any, list]
    :param graph: A graph, represented in any form - must be acceptable in graph_convertor
    :param output_type: take a list of nodes (vc) and convert it into the desired output type.
            is not given, output return as a list. must accept also None
    :param graph_convertor: convert a graph into Dict[Any, list] form. must accept as an input type(graph)
    :param k: maximum size of vc
    :param kwargs:
    :return:

    >>> sorted(vertex_cover(greedy, \
    [[False, False, True, True], [False, False, False, False], [True, False, False, True], [True, False, True ,False]], \
    graph_convertor=convert_matrix_to_list_graph))
    [0, 1]

    >>> sorted(vertex_cover(full_search, \
      [[False, False, True, True], [False, False, False, False], [True, False, False, True], [True, False, True ,False]], \
        graph_convertor=convert_matrix_to_list_graph))
    [0, 1]

    >>> vertex_cover(greedy, {'a': [], 'b':[], 'c':['d'], 'd':['c']}, lambda vc: len(vc), k=3)
    3

    >>> vertex_cover(greedy, {'a': [], 'b':[], 'c':['d'], 'd':['c']}, k=2)

    """

    dict_graph = graph_convertor(graph)

    result = algo(dict_graph, **kwargs)

    if 0 < k < len(result):
        result = None

    return output_type(result)


def main():
    doctest.testmod()


if __name__ == '__main__':
    main()
