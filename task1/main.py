from typing import Callable, Any, Iterable, List
from doctest import testmod
from collections.abc import Iterable as iterable


def sort_list(lst: list) -> list:
    return sorted(lst)


def add(x: int, y: float, z):
    return x + y + z


def safe_call(f: Callable, **kwargs) -> None:
    """
    call the function f if all arguments are of the right type.
    :raise TypeError: if not all arguments are of the right type
    :param f: callable
    :param kwargs: arguments for f
    :return:

    >>> safe_call(sort_list, lst = [1,3,6,1])
    [1, 1, 3, 6]

    >>> safe_call(sort_list, lst = [])
    []

    >>> safe_call(sort_list, lst = (1, 2, 3))
    Traceback (most recent call last):
    TypeError: wrong type in argument lst: expected <class 'list'>, but got <class 'tuple'>

    >>> safe_call(add, x=3, y=5.5, z=4)
    12.5

    >>> safe_call(add, x=3, y=9, z=1)
    Traceback (most recent call last):
    TypeError: wrong type in argument y: expected <class 'float'>, but got <class 'int'>

    """
    # check that all arguments are of the right type
    for arg_name, arg in kwargs.items():
        if arg_name in f.__annotations__.keys() and not type(arg) == f.__annotations__[arg_name]:   # check only annotated arguments
            raise TypeError(
                f'wrong type in argument {arg_name}: expected {f.__annotations__[arg_name]}, but got {type(arg)}')

    return f(**kwargs)


def four_neighbors(node):
    x, y = node
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def breadth_first_search(start: Any, end: Any, neighbor_function: Callable[[Any], List[Any]]):
    """
    run a BFS search from start to end, using neighbor_function
    :param start: starting node. can be any type, but must be consistent with neighbor_function
    :param end: end node. can be any type, but must be consistent with neighbor_function
    :param neighbor_function: function that return the neighbors of a node.
    :return: list representing the path from start to end

    >>> breadth_first_search((0, 0), (2, 2), four_neighbors)
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

    >>> breadth_first_search(7, 25, lambda n: [2*n, n-1])
    [7, 14, 13, 26, 25]

    >>> breadth_first_search(1, 10, lambda n: [n-1] if n>-13 else [])
    Traceback (most recent call last):
    Exception: path not found

    >>> breadth_first_search(('a', 'c', 56), ('a', 'c', 56), lambda d: d)
    [('a', 'c', 56)]
    """
    def trace_path() -> List[Any]:
        """
        trace the path from start to end. assume end is found
        :return: list of nodes representing the path
        """
        node = end
        path = [end]
        while not node == start:
            path.insert(0, visited_nodes[node])
            node = visited_nodes[node]
        return path

    # initiate
    queue = [start]
    visited_nodes = {start: None}
    current_node = None

    # main loop of BFS
    while queue:
        current_node = queue.pop(0)
        # if end is found
        if current_node == end:
            return trace_path()
        # add neighbors to queue
        for neighbor in neighbor_function(current_node):
            if neighbor not in visited_nodes.keys():
                visited_nodes.update({neighbor: current_node})
                queue.append(neighbor)

    raise Exception('path not found')


def print_sorted(lst: Iterable) -> None:
    """
    prints the items in lst, sorted (lexicographically). nested iterables are also printed in a sorted order
    :param lst: iterable to print sorted
    :return:

    >>> print_sorted({'a': 5, 'c': 6, 'b': [1, 3, 2, 4]})
    {'a': 5, 'b': [1, 2, 3, 4], 'c': 6}

    >>> print_sorted([])
    []

    >>> print_sorted({'g': 9,  78: 1, 19: ('fff', [5, -5, {}])})
    {'g': 9, 19: ([-5, 5, {}], 'fff'), 78: 1}
    """
    def deep_sort(x):
        """
        Sort the given input on all levels. non-iterable objects return as is. support lists, tuples, sets and dictionaries
        :param x: object to sort
        :return: a copy of x sorted on all levels
        """
        # non-iterable objects return as is
        if isinstance(x, str) or not isinstance(x, iterable):
            return x

        # lists, tuples, sets and dictionaries
        if isinstance(x, list):
            return sorted([deep_sort(item) for item in x], key=str)
        if isinstance(x, tuple):
            return tuple(sorted((deep_sort(item) for item in x), key=str))
        if isinstance(x, set):
            return set(sorted({deep_sort(item) for item in x}, key=str))
        if isinstance(x, dict):
            return dict(sorted({key: deep_sort(val) for key, val in x.items()}.items(), key=str))

    print(deep_sort(lst))   # print the sorted iterable


def main():
    testmod()


if __name__ == '__main__':
    main()

