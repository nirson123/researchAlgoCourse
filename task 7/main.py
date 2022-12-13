import numpy as np
import time
import matplotlib.pyplot as plt
import cvxpy
import networkx as nx
import itertools

def test_numpy_cvxpy(N_EQUATIONS_PER_SIZE: int = 50, size_limit: int = 200) -> (np.array, np.array):
    """
    test running time of linear equations solving of numpy and cvxpy.
    for each round, starting with 10 up to size_limit with jumps of 10,  n random equations with n variables are tested.
    to reduce variance, each round is preformed N_EQUATIONS_PER_SIZE times, and the result is the average.
    :param N_EQUATIONS_PER_SIZE: number of times each round is preformed
    :param size_limit: maximum nuber of equations to test
    :return: list of numpy time results,  list of cvxpy time results
    """
    times_numpy = []
    times_cvxpy = []

    # n equations with n variables
    for n in range(10, size_limit + 1, 10):

        total_time_numpy = 0
        total_time_cvxpy = 0

        for _ in range(N_EQUATIONS_PER_SIZE):

            # random coefficients (between 1 and 1000)
            left_hand_side = np.random.uniform(1, 1000, (n, n))
            right_hand_side = np.random.uniform(1, 1000, n)

            # numpy
            start_time = time.time()
            np.linalg.solve(left_hand_side, right_hand_side)
            end_time = time.time()
            total_time_numpy += (end_time - start_time)

            # cvxpy
            vars = cvxpy.Variable(n)
            constraints = [left_hand_side @ vars == right_hand_side]
            obj = cvxpy.Minimize(vars[0])   # doesn't matter - only care about constraints
            problem = cvxpy.Problem(obj, constraints)
            start_time = time.time()
            problem.solve()
            end_time = time.time()
            total_time_cvxpy += (end_time - start_time)

        # average over all tries
        times_numpy.append(total_time_numpy / N_EQUATIONS_PER_SIZE)
        times_cvxpy.append(total_time_cvxpy / N_EQUATIONS_PER_SIZE)

    return np.array(times_numpy), np.array(times_cvxpy)


def Q1():
    """
    answer to question 1
    :return:
    """
    # get time results
    results_numpy, results_scipy = test_numpy_cvxpy(N_EQUATIONS_PER_SIZE=100, size_limit=250)

    # plot
    plt.plot(np.arange(10, 251, 10), results_numpy, label='numpy')
    plt.plot(np.arange(10, 251, 10), results_scipy, label='cvxpy')
    plt.legend()
    plt.xlabel('Number of equations / variables')
    plt.ylabel('time(Sec.)')
    plt.show()


def approximation_test(p_values, min_nodes: int = 5, max_nodes: int = 20, n_tries: int = 10) -> np.ndarray:
    """
    test the approximation quality of networkx max_clique approximation algorithm on random graphs
    worst case factor is |V| / log^2(|v|) in the limit.
    :param p_values: test on G(n,p) random graphs. set of p values to test
    :param min_nodes: minimum n to test for each p
    :param max_nodes: maximum n to test for each p
    :param n_tries:  to reduce variance, each pair on (n,p) is tests multiple times and the result os the average
    :return: approximation ratios for each (p,n) pair.
    """
    results = []
    for p in p_values:
        current_p_results = []

        # test different graphs sizes
        for n in range(min_nodes, max_nodes + 1):
            total = 0
            for _ in range(n_tries):     # take average of multiple try, to reduce variance
                graph = nx.gnp_random_graph(n, p)

                app = nx.approximation.max_clique(graph)            # approximation
                real = nx.max_weight_clique(graph, weight=None)     # real result using full search

                total += len(app) / len(real)   # ratio

            current_p_results.append(total / n_tries)

        results.append(np.array(current_p_results))

    return np.array(results)


def Q2():
    """
    answer to question 2
    :return:
    """

    max_nodes = 100
    # get approximation ratios results
    results = approximation_test(np.arange(0.1, 1, 0.1), min_nodes=5, max_nodes=max_nodes)

    # plot
    fig, ax = plt.subplots(3, 3)
    for i, j in itertools.product(range(3), range(3)):
        ax[i, j].set_title(f'p = {np.round((i*3 + j + 1) * 0.1, decimals=2)}')
        ax[i, j].plot(range(5, max_nodes + 1), results[i*3 + j])
        ax[i, j].set_xlabel('n')
        ax[i, j].set_ylabel('approx. ratio')
        ax[i, j].set_xticks(range(5, max_nodes + 1, 10))

        # theoretical approximation ratio
        app_ratio = np.arange(5, max_nodes + 1, 1) / ((np.log(np.arange(5, max_nodes + 1, 1))) ** 2)
        ax[i, j].plot(range(5, max_nodes + 1), app_ratio)

    plt.tight_layout()
    plt.show()


def main():
    Q1()
    Q2()


if __name__ == '__main__':
    main()
