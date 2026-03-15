from src.calculator import backtrack
from src.calculator import binary_search
from src.calculator import combinations
from src.calculator import detect_cycle_directed
from src.calculator import dfs
from src.calculator import dijkstra
from src.calculator import edit_distance
from src.calculator import evaluate_rpn
from src.calculator import fibonacci
from src.calculator import flatten_list
from src.calculator import gcd
from src.calculator import generate_subsets
from src.calculator import group_anagrams
from src.calculator import int_to_roman
from src.calculator import is_pal
from src.calculator import is_prime
from src.calculator import knapsack
from src.calculator import lcm, gcd
from src.calculator import longest_common_subsequence
from src.calculator import longest_unique_substring
from src.calculator import majority_element
from src.calculator import matrix_multiply
from src.calculator import merge_intervals
from src.calculator import palindrome_partitions
from src.calculator import permutations
from src.calculator import prime_factors
from src.calculator import roman_to_int
from src.calculator import shortest_path_bfs
from src.calculator import sliding_window_max
from src.calculator import top_k_frequent
from src.calculator import transpose_matrix
from typing import Dict, List
from typing import Dict, List, Tuple
from typing import List
from typing import List, Tuple
import pytest

def test_fibonacci():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(5) == 5
    with pytest.raises(ValueError):
        fibonacci(-1)
    with pytest.raises(ValueError):
        fibonacci(10**100)
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765



def test_is_prime():
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(4) == False
    assert is_prime(5) == True
    assert is_prime(6) == False
    assert is_prime(7) == True
    assert is_prime(8) == False
    assert is_prime(9) == False
    assert is_prime(10) == False
    with pytest.raises(TypeError):
        is_prime('a')
    with pytest.raises(TypeError):
        is_prime(1.5)



def test_prime_factors():
    assert prime_factors(1) == []
    assert prime_factors(2) == [2]
    assert prime_factors(3) == [3]
    assert prime_factors(4) == [2, 2]
    assert prime_factors(5) == [5]
    assert prime_factors(6) == [2, 3]
    assert prime_factors(7) == [7]
    assert prime_factors(8) == [2, 2, 2]
    assert prime_factors(9) == [3, 3]
    assert prime_factors(10) == [2, 5]
    with pytest.raises(TypeError):
        prime_factors("a")
    with pytest.raises(TypeError):
        prime_factors(1.5)



def test_gcd():
    assert gcd(10, 15) == 5
    assert gcd(0, 10) == 10
    assert gcd(10, 0) == 10
    assert gcd(0, 0) == 0
    with pytest.raises(TypeError):
        gcd('a', 10)
    with pytest.raises(TypeError):
        gcd(10, 'b')



def test_lcm():
    assert lcm(2, 3) == 6
    assert lcm(7, 11) == 77
    assert lcm(0, 10) == 10
    assert lcm(10, 0) == 10
    with pytest.raises(TypeError):
        lcm('a', 3)
    with pytest.raises(TypeError):
        lcm(2, 'b')
    with pytest.raises(TypeError):
        lcm('a', 'b')
    assert lcm(-2, 3) == 6
    assert lcm(2, -3) == 6
    assert lcm(-2, -3) == 6



def test_flatten_list():
    assert flatten_list([1, 2, [3, 4], [5, [6, 7]]]) == [1, 2, 3, 4, 5, 6, 7]
    assert flatten_list([]) == []
    assert flatten_list([1, 2, 3]) == [1, 2, 3]
    assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
    with pytest.raises(TypeError):
        flatten_list("hello")
    with pytest.raises(TypeError):
        flatten_list(123)



def test_longest_common_subsequence():
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("abc", "abcd") == 1
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("", "abc") == 0
    assert longest_common_subsequence("abc", "") == 0
    assert longest_common_subsequence("abc", "abcde") == 3
    with pytest.raises(TypeError):
        longest_common_subsequence(123, "abc")
    with pytest.raises(TypeError):
        longest_common_subsequence("abc", 123)



def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "hello") == 5
    assert edit_distance("hello", "") == 5
    assert edit_distance("hello", "world") == 4
    assert edit_distance("hello", "hello") == 0
    with pytest.raises(TypeError):
        edit_distance(123, "hello")
    with pytest.raises(TypeError):
        edit_distance("hello", 123)
    with pytest.raises(TypeError):
        edit_distance(123, 456)



def test_knapsack():
    assert knapsack([1, 2, 3], [10, 20, 30], 5) == 30
    assert knapsack([1, 2, 3], [10, 20, 30], 10) == 40
    assert knapsack([1, 2, 3], [10, 20, 30], 0) == 0
    with pytest.raises(TypeError):
        knapsack('a', [10, 20, 30], 5)
    with pytest.raises(TypeError):
        knapsack([1, 2, 3], 'a', 5)
    with pytest.raises(TypeError):
        knapsack([1, 2, 3], [10, 20, 30], 'a')
    assert knapsack([1, 2, 3], [10, 20, 30], -1) == 0
    assert knapsack([], [10, 20, 30], 5) == 0
    assert knapsack([1, 2, 3], [], 5) == 0



def test_permutations():
    assert permutations([1, 2, 3]) == [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    assert permutations([]) == []
    assert permutations([1]) == [(1,)]
    with pytest.raises(TypeError):
        permutations('123')
    with pytest.raises(TypeError):
        permutations(123)



def test_combinations():
    # Test with a list of integers
    assert combinations([1, 2, 3], 2) == [(1, 2), (1, 3), (2, 3)]

    # Test with an empty list
    assert combinations([], 2) == []

    # Test with a list of non-integer values
    with pytest.raises(TypeError):
        combinations([1, 2, '3'], 2)

    # Test with a negative value for r
    with pytest.raises(ValueError):
        combinations([1, 2, 3], -1)

    # Test with a value of r greater than the length of the list
    with pytest.raises(ValueError):
        combinations([1, 2, 3], 4)



def test_binary_search():
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    assert binary_search([1, 2, 3, 4, 5], 6) == -1
    assert binary_search([1, 2, 3, 4, 5], 1) == 0
    with pytest.raises(TypeError):
        binary_search("12345", 3)
    with pytest.raises(TypeError):
        binary_search([1, 2, 3, 4, 5], "3")
    assert binary_search([], 3) == -1
    assert binary_search([1], 1) == 0
    assert binary_search([1, 2, 3, 4, 5], 5) == 4



def test_merge_intervals():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
    assert merge_intervals([[1,3],[5,8],[8,10]]) == [[1,3],[5,10]]
    assert merge_intervals([]) == []
    assert merge_intervals([[1,2]]) == [[1,2]]
    with pytest.raises(TypeError):
        merge_intervals("not a list")
    with pytest.raises(TypeError):
        merge_intervals([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])



def test_top_k_frequent():
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 3) == [1, 2, 3]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 1) == [1]
    assert top_k_frequent([], 1) == []
    with pytest.raises(TypeError):
        top_k_frequent("123", 1)
    with pytest.raises(TypeError):
        top_k_frequent([1, 2, 3], "a")
    with pytest.raises(ValueError):
        top_k_frequent([1, 2, 3], -1)



def test_group_anagrams():
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert group_anagrams(["hello", "world"]) == [["hello"], ["world"]]
    assert group_anagrams([""]) == [[""]]
    with pytest.raises(TypeError):
        group_anagrams(123)
    with pytest.raises(TypeError):
        group_anagrams("hello")



def test_palindrome_partitions():
    assert palindrome_partitions("a") == [["a"]]
    assert palindrome_partitions("abba") == [["abba"]]
    assert palindrome_partitions("abcba") == [["abcba"]]
    assert palindrome_partitions("abc") == []
    assert palindrome_partitions("") == [[]]
    with pytest.raises(TypeError):
        palindrome_partitions(123)
    with pytest.raises(TypeError):
        palindrome_partitions(None)



def test_is_pal():
    assert is_pal(121) == True
    assert is_pal(123) == False
    assert is_pal(12321) == True
    assert is_pal(-121) == False
    with pytest.raises(TypeError):
        is_pal("121")



def test_backtrack():
    s = "abba"
    result = []
    backtrack(0, [])
    assert len(result) == 2
    assert result == [["a", "b", "b", "a"], ["b", "b", "a", "a"]]

    s = "abc"
    result = []
    backtrack(0, [])
    assert len(result) == 0

    s = "aaa"
    result = []
    backtrack(0, [])
    assert len(result) == 1
    assert result == [["a", "a", "a"]]

    s = ""
    result = []
    backtrack(0, [])
    assert len(result) == 1
    assert result == [[]]

    s = "a"
    result = []
    backtrack(0, [])
    assert len(result) == 1
    assert result == [["a"]]

    s = "ab"
    result = []
    backtrack(0, [])
    assert len(result) == 1
    assert result == [["a", "b"]]

    s = "aabb"
    result = []
    backtrack(0, [])
    assert len(result) == 2
    assert result == [["a", "a", "b", "b"], ["a", "b", "a", "b"]]




def test_transpose_matrix():
    matrix = [[1, 2, 3], [4, 5, 6]]
    expected_result = [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix(matrix) == expected_result

    matrix = [[1, 2], [3, 4], [5, 6]]
    expected_result = [[1, 3, 5], [2, 4, 6]]
    assert transpose_matrix(matrix) == expected_result

    matrix = [[1]]
    expected_result = [[1]]
    assert transpose_matrix(matrix) == expected_result

    matrix = []
    expected_result = []
    assert transpose_matrix(matrix) == expected_result

    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    expected_result = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    assert transpose_matrix(matrix) == expected_result

    matrix = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(TypeError):
        transpose_matrix("matrix")

    matrix = [[1, 2, 3], [4, 5, 6]]
    with pytest.raises(TypeError):
        transpose_matrix(123)



def test_shortest_path_bfs():
    graph = {
        0: [1, 2],
        1: [2],
        2: [3],
        3: [4],
        4: []
    }
    assert shortest_path_bfs(graph, 0, 4) == 3
    assert shortest_path_bfs(graph, 0, 0) == 0
    assert shortest_path_bfs(graph, 4, 0) == -1
    with pytest.raises(KeyError):
        shortest_path_bfs(graph, 5, 4)
    with pytest.raises(KeyError):
        shortest_path_bfs(graph, 0, 5)
    graph = {
        0: [1, 2],
        1: [2],
        2: [3],
        3: [4],
        4: [0]
    }
    assert shortest_path_bfs(graph, 0, 4) == 2
    graph = {
        0: [1, 2],
        1: [2],
        2: [3],
        3: [4],
        4: []
    }
    assert shortest_path_bfs(graph, 0, 4) == 3
    graph = {
        0: [1, 2],
        1: [2],
        2: [3],
        3: [4],
        4: []
    }
    assert shortest_path_bfs(graph, 0, 4) == 3



def test_detect_cycle_directed():
    # Test case 1: No cycle in the graph
    graph1 = {0: [1, 2], 1: [2], 2: []}
    assert not detect_cycle_directed(graph1)

    # Test case 2: Cycle in the graph
    graph2 = {0: [1], 1: [2], 2: [0]}
    assert detect_cycle_directed(graph2)

    # Test case 3: Empty graph
    graph3 = {}
    assert not detect_cycle_directed(graph3)

    # Test case 4: Graph with a single node
    graph4 = {0: []}
    assert not detect_cycle_directed(graph4)

    # Test case 5: Graph with multiple disconnected components
    graph5 = {0: [1], 1: [2], 2: [], 3: [4], 4: []}
    assert not detect_cycle_directed(graph5)

    # Test case 6: Graph with a cycle and multiple disconnected components
    graph6 = {0: [1], 1: [2], 2: [0], 3: [4], 4: []}
    assert detect_cycle_directed(graph6)

    # Test case 7: Graph with a cycle and a node not in the cycle
    graph7 = {0: [1], 1: [2], 2: [0], 3: [4], 4: [3]}
    assert detect_cycle_directed(graph7)



def test_dfs():
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D'],
        'C': ['A', 'D'],
        'D': ['B', 'C']
    }
    stack = set()
    visited = set()
    assert dfs('A', graph, stack, visited) == True
    assert dfs('D', graph, stack, visited) == True
    assert dfs('E', graph, stack, visited) == False
    with pytest.raises(KeyError):
        dfs('A', {}, stack, visited)
    with pytest.raises(TypeError):
        dfs('A', graph, 'stack', visited)
    with pytest.raises(TypeError):
        dfs('A', graph, stack, 'visited')



def test_dijkstra():
    graph = {0: [(1, 4), (2, 2)], 1: [(2, 1), (3, 5)], 2: [(3, 8)], 3: []}
    assert dijkstra(graph, 0) == {0: 0, 1: 4, 2: 2, 3: 9}

    graph = {0: [(1, 1), (2, 1)], 1: [(2, 1)], 2: []}
    assert dijkstra(graph, 0) == {0: 0, 1: 1, 2: 2}

    graph = {0: [(1, 1), (2, 1)], 1: [(2, 1)], 2: []}
    with pytest.raises(KeyError):
        dijkstra(graph, 3)

    graph = {0: [(1, -1)], 1: []}
    with pytest.raises(ValueError):
        dijkstra(graph, 0)

    graph = {0: [(1, 1)], 1: []}
    assert dijkstra(graph, 0) == {0: 0, 1: 1}

    graph = {0: [(1, 1)], 1: []}
    assert dijkstra(graph, 1) == {0: math.inf, 1: 0}




def test_majority_element():
    assert majority_element([3, 2, 3]) == 3
    assert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2
    assert majority_element([1]) == 1
    with pytest.raises(ValueError):
        majority_element([])
    with pytest.raises(ValueError):
        majority_element([1, 2, 3, 4, 5, 6, 7, 8, 9])



def test_roman_to_int():
    assert roman_to_int('I') == 1
    assert roman_to_int('V') == 5
    assert roman_to_int('X') == 10
    assert roman_to_int('L') == 50
    assert roman_to_int('C') == 100
    assert roman_to_int('D') == 500
    assert roman_to_int('M') == 1000
    assert roman_to_int('II') == 2
    assert roman_to_int('III') == 3
    assert roman_to_int('IV') == 4
    assert roman_to_int('IX') == 9
    assert roman_to_int('XL') == 40
    assert roman_to_int('XC') == 90
    assert roman_to_int('CD') == 400
    assert roman_to_int('CM') == 900
    assert roman_to_int('MMXXI') == 2021
    with pytest.raises(KeyError):
        roman_to_int('A')
    with pytest.raises(KeyError):
        roman_to_int('IIV')



def test_int_to_roman():
    assert int_to_roman(1) == "I"
    assert int_to_roman(4) == "IV"
    assert int_to_roman(5) == "V"
    assert int_to_roman(9) == "IX"
    assert int_to_roman(13) == "XIII"
    assert int_to_roman(44) == "XLIV"
    assert int_to_roman(1000) == "M"
    with pytest.raises(TypeError):
        int_to_roman("1")
    with pytest.raises(ValueError):
        int_to_roman(-1)



def test_evaluate_rpn():
    assert evaluate_rpn(["5", "3", "+"]) == 8
    assert evaluate_rpn(["5", "3", "-"]) == 2
    assert evaluate_rpn(["5", "3", "*"]) == 15
    assert evaluate_rpn(["5", "3", "/"]) == 1
    assert evaluate_rpn(["5", "3", "+", "4", "*"]) == 35
    assert evaluate_rpn(["5", "3", "+", "4", "/"]) == 6
    with pytest.raises(IndexError):
        evaluate_rpn(["5"])
    with pytest.raises(IndexError):
        evaluate_rpn(["5", "3"])
    with pytest.raises(IndexError):
        evaluate_rpn(["5", "3", "+", "4"])



def test_generate_subsets():
    assert generate_subsets([1, 2, 3]) == [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    assert generate_subsets([]) == [[]]
    assert generate_subsets([1]) == [[], [1]]
    with pytest.raises(TypeError):
        generate_subsets('123')
    with pytest.raises(TypeError):
        generate_subsets(123)



def test_longest_unique_substring():
    assert longest_unique_substring("abcabcbb") == 3
    assert longest_unique_substring("bbbbbb") == 1
    assert longest_unique_substring("pwwkew") == 3
    assert longest_unique_substring("") == 0
    assert longest_unique_substring("abcdefghijklmnopqrstuvwxyz") == 26
    with pytest.raises(TypeError):
        longest_unique_substring(123)
    with pytest.raises(TypeError):
        longest_unique_substring(None)



def test_matrix_multiply():
    assert matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
    assert matrix_multiply([[1, 2], [3, 4]], [[5], [7]]) == [[17], [39]]
    assert matrix_multiply([[1, 2], [3, 4]], [[5, 6]]) == [[19, 22]]
    assert matrix_multiply([[1, 2], [3, 4]], []) == []
    assert matrix_multiply([], [[5, 6], [7, 8]]) == []
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 9]])
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[5, 6], [7, 8]])
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2], [3, 4]], [[5, 6]])
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2]], [[5, 6], [7, 8]])
    with pytest.raises(ValueError):
        matrix_multiply([], [[5, 6], [7, 8]])
    with pytest.raises(ValueError):
        matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8, 9]])

def test_sliding_window_max():
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3) == [3, 3, 5, 5, 6, 7]
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 1) == [1, 3, -1, -3, 5, 3, 6, 7]
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 8) == [7]
    assert sliding_window_max([], 3) == []
    with pytest.raises(ValueError):
        sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 0)
    with pytest.raises(ValueError):
        sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], -3)
    with pytest.raises(ValueError):
        sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3)
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 2) == [3, 3, 5, 5]
    assert sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 4) == [3, 5, 5, 6]
