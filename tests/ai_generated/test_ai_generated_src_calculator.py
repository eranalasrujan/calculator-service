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
    assert is_prime(11) == True
    assert is_prime(12) == False
    assert is_prime(13) == True
    assert is_prime(14) == False
    assert is_prime(15) == False
    assert is_prime(16) == False
    assert is_prime(17) == True
    assert is_prime(18) == False
    assert is_prime(19) == True
    assert is_prime(20) == False

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
    with pytest.raises(TypeError):
        prime_factors(None)
    assert prime_factors(-1) == []
    assert prime_factors(0) == []

def test_gcd():
    assert gcd(10, 15) == 5
    assert gcd(0, 10) == 10
    assert gcd(10, 0) == 10
    assert gcd(0, 0) == 0
    with pytest.raises(TypeError):
        gcd('a', 10)
    with pytest.raises(TypeError):
        gcd(10, 'b')
    with pytest.raises(TypeError):
        gcd('a', 'b')
    assert gcd(-10, 15) == 5
    assert gcd(10, -15) == 5
    assert gcd(-10, -15) == 5

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
    assert lcm(10, 15) == 30
    assert lcm(20, 30) == 60
    assert lcm(100, 200) == 2000

def test_flatten_list():
    assert flatten_list([1, 2, [3, 4], [5, [6, 7]]]) == [1, 2, 3, 4, 5, 6, 7]
    assert flatten_list([]) == []
    assert flatten_list([1, 2, 3]) == [1, 2, 3]
    assert flatten_list([1, [2, 3], 4]) == [1, 2, 3, 4]
    assert flatten_list([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]
    assert flatten_list([1, [2, [3, [4, 5]]], 6]) == [1, 2, 3, 4, 5, 6]
    with pytest.raises(TypeError):
        flatten_list("hello")
    with pytest.raises(TypeError):
        flatten_list(123)
    with pytest.raises(TypeError):
        flatten_list(None)
    with pytest.raises(TypeError):
        flatten_list(True)
    with pytest.raises(TypeError):
        flatten_list(1.5)

def test_longest_common_subsequence():
    assert longest_common_subsequence("abc", "def") == 0
    assert longest_common_subsequence("abc", "abcd") == 1
    assert longest_common_subsequence("abc", "abc") == 3
    assert longest_common_subsequence("", "abc") == 0
    assert longest_common_subsequence("abc", "") == 0
    assert longest_common_subsequence("abc", "abcde") == 3
    assert longest_common_subsequence("abcdef", "abc") == 3
    assert longest_common_subsequence("abc", "abcdef") == 3
    assert longest_common_subsequence("abc", "abcxyz") == 3
    assert longest_common_subsequence("abc", "xyzabc") == 3
    with pytest.raises(TypeError):
        longest_common_subsequence("a", 123)
    with pytest.raises(TypeError):
        longest_common_subsequence(123, "abc")

def test_edit_distance():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "hello") == 5
    assert edit_distance("hello", "") == 5
    assert edit_distance("hello", "world") == 4
    assert edit_distance("a", "b") == 1
    assert edit_distance("a", "a") == 0
    with pytest.raises(TypeError):
        edit_distance("a", 123)
    with pytest.raises(TypeError):
        edit_distance(123, "a")
    with pytest.raises(TypeError):
        edit_distance("a", "b", 123)

def test_knapsack():
    assert knapsack([1, 2, 3], [10, 20, 30], 5) == 30
    assert knapsack([1, 2, 3], [10, 20, 30], 10) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 0) == 0
    assert knapsack([], [10, 20, 30], 5) == 0
    assert knapsack([1, 2, 3], [], 5) == 0
    with pytest.raises(TypeError):
        knapsack('a', [10, 20, 30], 5)
    with pytest.raises(TypeError):
        knapsack([1, 2, 3], 'b', 5)
    with pytest.raises(TypeError):
        knapsack([1, 2, 3], [10, 20, 30], 'c')
    assert knapsack([1, 2, 3], [10, 20, 30], -5) == 0
    assert knapsack([1, 2, 3], [10, 20, 30], 5) == 30
    assert knapsack([1, 2, 3], [10, 20, 30], 15) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 20) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 25) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 30) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 35) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 40) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 45) == 60
    assert knapsack([1, 2, 3], [10, 20, 30], 50) == 60

def test_permutations():
    assert permutations([1, 2, 3]) == [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    assert permutations([]) == [()]
    assert permutations([1]) == [(1,)]
    assert permutations([1, 2]) == [(1, 2), (2, 1)]
    with pytest.raises(TypeError):
        permutations("hello")
    with pytest.raises(TypeError):
        permutations(123)
    with pytest.raises(TypeError):
        permutations(1.5)
    with pytest.raises(TypeError):
        permutations([1, 2, 3.4])
    with pytest.raises(TypeError):
        permutations([1, 2, '3'])

def test_combinations():
    assert combinations([1, 2, 3], 2) == [(1, 2), (1, 3), (2, 3)]
    assert combinations([1, 2, 3], 1) == [(1,), (2,), (3,)]
    assert combinations([1, 2, 3], 3) == [(1, 2, 3)]
    assert combinations([1, 2, 3], 4) == []
    with pytest.raises(TypeError):
        combinations("hello", 2)
    with pytest.raises(TypeError):
        combinations(123, 2)
    with pytest.raises(TypeError):
        combinations([1, 2, 3], "hello")
    with pytest.raises(TypeError):
        combinations([1, 2, 3], 2.5)
    with pytest.raises(ValueError):
        combinations([1, 2, 3], -1)

def test_binary_search():
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert binary_search(arr, 5) == 4
    assert binary_search(arr, 10) == -1
    assert binary_search(arr, 1) == 0
    assert binary_search(arr, 9) == 8
    with pytest.raises(TypeError):
        binary_search("hello", 5)
    with pytest.raises(TypeError):
        binary_search([1, 2, 3], "hello")
    with pytest.raises(TypeError):
        binary_search(123, 5)
    with pytest.raises(TypeError):
        binary_search([1, 2, 3], 123)

def test_merge_intervals():
    assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]
    assert merge_intervals([[1,4],[4,5]]) == [[1,5]]
    assert merge_intervals([[1,3],[5,8],[8,10]]) == [[1,3],[5,10]]
    assert merge_intervals([]) == []
    assert merge_intervals([[1,2]]) == [[1,2]]
    with pytest.raises(TypeError):
        merge_intervals("hello")
    with pytest.raises(TypeError):
        merge_intervals(123)
    assert merge_intervals([[1,2],[2,3]]) == [[1,3]]
    assert merge_intervals([[1,3],[2,3]]) == [[1,3]]
    assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]
    assert merge_intervals([[1,3],[2,3],[3,4]]) == [[1,4]]

def test_top_k_frequent():
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 1) == [1]
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 3) == [1, 2, 3]
    with pytest.raises(TypeError):
        top_k_frequent("hello", 2)
    with pytest.raises(TypeError):
        top_k_frequent([1, 2, 3], "a")
    with pytest.raises(TypeError):
        top_k_frequent([1, 2, 3], 2.5)
    with pytest.raises(TypeError):
        top_k_frequent("hello", 2.5)
    assert top_k_frequent([], 2) == []
    assert top_k_frequent([1, 2, 3], 0) == []

def test_group_anagrams():
    assert group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]) == [["ate", "eat", "tea"], ["nat", "tan"], ["bat"]]
    assert group_anagrams(["a"]) == [["a"]]
    assert group_anagrams([]) == []
    with pytest.raises(TypeError):
        group_anagrams("hello")
    with pytest.raises(TypeError):
        group_anagrams(123)
    assert group_anagrams(["hello", "world"]) == [["hello"], ["world"]]
    assert group_anagrams(["abc", "bca", "cab"]) == [["abc", "bca", "cab"]]
    assert group_anagrams(["abc", "def", "ghi"]) == [["abc"], ["def"], ["ghi"]]

def test_palindrome_partitions():
    assert palindrome_partitions("a") == [["a"]]
    assert palindrome_partitions("abba") == [["a", "bba"], ["abba"]]
    assert palindrome_partitions("abcba") == [["a", "bcba"], ["ab", "cba"], ["abcba"]]
    assert palindrome_partitions("racecar") == [["r", "aceca"], ["ra", "ceca"], ["rac", "eca"], ["racecar"]]
    assert palindrome_partitions("") == [[]]
    with pytest.raises(TypeError):
        palindrome_partitions(123)
    with pytest.raises(TypeError):
        palindrome_partitions("hello")

def test_is_pal():
    assert is_pal(12321) == True
    assert is_pal(123456) == False
    assert is_pal(123321) == True
    assert is_pal(0) == True
    assert is_pal(-12321) == True
    with pytest.raises(TypeError):
        is_pal('12321')
    with pytest.raises(TypeError):
        is_pal(123.21)

def test_backtrack():
    assert backtrack("a", []) == []
    assert backtrack("ab", []) == ["a", "b"]
    assert backtrack("abc", []) == ["a", "b", "c"]
    assert backtrack("abba", []) == ["a", "b", "b", "a"]
    with pytest.raises(ValueError):
        backtrack("abc", [1, 2, 3])
    with pytest.raises(ValueError):
        backtrack("abc", "hello")

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

def test_transpose_matrix():
    assert transpose_matrix([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix([[1]]) == [[1]]
    assert transpose_matrix([]) == []
    with pytest.raises(TypeError):
        transpose_matrix("hello")
    with pytest.raises(TypeError):
        transpose_matrix(123)
    with pytest.raises(TypeError):
        transpose_matrix([1, 2, "hello"])
    with pytest.raises(TypeError):
        transpose_matrix([1, 2, 3.5])
    with pytest.raises(ValueError):
        transpose_matrix([[1, 2], [3]])
    with pytest.raises(ValueError):
        transpose_matrix([[1, 2, 3], [4, 5]])

def test_shortest_path_bfs():
    graph = {0: [1, 2], 1: [2], 2: [3], 3: []}
    assert shortest_path_bfs(graph, 0, 3) == 2
    assert shortest_path_bfs(graph, 1, 3) == 2
    assert shortest_path_bfs(graph, 2, 3) == 1
    assert shortest_path_bfs(graph, 3, 3) == 0
    with pytest.raises(ValueError):
        shortest_path_bfs(graph, 0, 4)
    with pytest.raises(ValueError):
        shortest_path_bfs(graph, 4, 3)
    with pytest.raises(ValueError):
        shortest_path_bfs({}, 0, 3)
    with pytest.raises(ValueError):
        shortest_path_bfs(graph, 0, 0)
    with pytest.raises(ValueError):
        shortest_path_bfs(graph, 'a', 3)

def test_detect_cycle_directed():
    graph = {1: [2, 3], 2: [4], 3: [4], 4: [1]}
    assert detect_cycle_directed(graph) == True
    graph = {1: [2], 2: [3], 3: [4], 4: [5]}
    assert detect_cycle_directed(graph) == False
    graph = {1: [2], 2: [3], 3: [4], 4: [1, 5]}
    assert detect_cycle_directed(graph) == True
    graph = {}
    assert detect_cycle_directed(graph) == False
    with pytest.raises(TypeError):
        detect_cycle_directed("hello")
    with pytest.raises(TypeError):
        detect_cycle_directed(123)

def test_dfs():
    graph = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    stack = set()
    visited = set()
    assert dfs('A', graph, stack, visited) == True
    assert dfs('F', graph, stack, visited) == True
    assert dfs('G', graph, stack, visited) == False
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
    with pytest.raises(ValueError):
        dijkstra(graph, 3)
    graph = {0: [(1, 1), (2, 1)], 1: [(2, 1)], 2: []}
    with pytest.raises(ValueError):
        dijkstra(graph, -1)
    graph = {0: [(1, 1), (2, 1)], 1: [(2, 1)], 2: []}
    with pytest.raises(ValueError):
        dijkstra(graph, 'a')
    graph = {0: [(1, 1), (2, 1)], 1: [(2, 1)], 2: []}
    with pytest.raises(ValueError):
        dijkstra(graph, 1.5)

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

def test_majority_element():
    assert majority_element([1, 1, 2, 1, 1, 3]) == 1
    assert majority_element([1, 2, 3, 4, 5]) == None
    assert majority_element([1, 1, 1, 2, 2]) == 1
    assert majority_element([]) == None
    with pytest.raises(TypeError):
        majority_element("hello")
    with pytest.raises(TypeError):
        majority_element(123)
    with pytest.raises(TypeError):
        majority_element([1, 2, "3"])
    with pytest.raises(TypeError):
        majority_element([1, 2, 3.4])

def test_roman_to_int():
    assert roman_to_int("I") == 1
    assert roman_to_int("V") == 5
    assert roman_to_int("X") == 10
    assert roman_to_int("L") == 50
    assert roman_to_int("C") == 100
    assert roman_to_int("D") == 500
    assert roman_to_int("M") == 1000
    assert roman_to_int("III") == 3
    assert roman_to_int("IV") == 4
    assert roman_to_int("IX") == 9
    assert roman_to_int("LVIII") == 58
    assert roman_to_int("MCMXCIV") == 1994
    with pytest.raises(TypeError):
        roman_to_int(123)
    with pytest.raises(ValueError):
        roman_to_int("")
    with pytest.raises(ValueError):
        roman_to_int("abc")
    with pytest.raises(ValueError):
        roman_to_int("IIII")

def test_int_to_roman():
    assert int_to_roman(1) == "I"
    assert int_to_roman(4) == "IV"
    assert int_to_roman(5) == "V"
    assert int_to_roman(9) == "IX"
    assert int_to_roman(10) == "X"
    assert int_to_roman(40) == "XL"
    assert int_to_roman(50) == "L"
    assert int_to_roman(90) == "XC"
    assert int_to_roman(100) == "C"
    assert int_to_roman(400) == "CD"
    assert int_to_roman(500) == "D"
    assert int_to_roman(900) == "CM"
    assert int_to_roman(1000) == "M"
    with pytest.raises(ValueError):
        int_to_roman(-1)
    with pytest.raises(ValueError):
        int_to_roman(0)
    with pytest.raises(ValueError):
        int_to_roman(1001)
    with pytest.raises(TypeError):
        int_to_roman("a")
    with pytest.raises(TypeError):
        int_to_roman(1.5)

def test_generate_subsets():
    assert generate_subsets([1, 2, 3]) == [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]
    assert generate_subsets([]) == [[]]
    assert generate_subsets([1]) == [[], [1]]
    assert generate_subsets([1, 2]) == [[], [1], [2], [1, 2]]
    with pytest.raises(TypeError):
        generate_subsets("hello")
    with pytest.raises(TypeError):
        generate_subsets(123)
    with pytest.raises(TypeError):
        generate_subsets(1.5)
    assert generate_subsets([1, 2, 2]) == [[], [1], [2], [1, 2], [2], [1, 2], [2, 2], [1, 2, 2]]
    assert generate_subsets([1, 2, 3, 4]) == [[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3], [4], [1, 4], [2, 4], [1, 2, 4], [3, 4], [1, 3, 4], [2, 3, 4], [1, 2, 3, 4]]

def test_word_frequency():
    assert word_frequency("hello world") == {'hello': 1, 'world': 1}
    assert word_frequency("hello world world") == {'hello': 1, 'world': 2}
    assert word_frequency("") == {}
    assert word_frequency("   ") == {}
    assert word_frequency("hello") == {'hello': 1}
    with pytest.raises(TypeError):
        word_frequency(123)
    with pytest.raises(TypeError):
        word_frequency("hello world".upper())
    with pytest.raises(TypeError):
        word_frequency("hello world".lower().split())

def test_longest_unique_substring():
    assert longest_unique_substring("abcabcbb") == 3
    assert longest_unique_substring("bbbbbb") == 1
    assert longest_unique_substring("pwwkew") == 3
    assert longest_unique_substring("") == 0
    assert longest_unique_substring("a") == 1
    with pytest.raises(TypeError):
        longest_unique_substring(123)
    with pytest.raises(TypeError):
        longest_unique_substring("hello")
    assert longest_unique_substring("abcdefghijklmnopqrstuvwxyz") == 26
    assert longest_unique_substring("abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz") == 26
    assert longest_unique_substring("aabbcc") == 2
    assert longest_unique_substring("abcabc") == 3
