from src.dsa import binary_search
from src.dsa import is_valid_parentheses
from src.dsa import longest_unique_substring
from src.dsa import merge_intervals
import pytest

def test_is_valid_parentheses():
    assert is_valid_parentheses("()") == True
    assert is_valid_parentheses("()[]{}") == True
    assert is_valid_parentheses("(]") == False
    assert is_valid_parentheses("([)]") == False
    assert is_valid_parentheses("{[]}") == True
    assert is_valid_parentheses("") == True
    with pytest.raises(TypeError):
        is_valid_parentheses(123)

def test_binary_search():
    # Test case 1: Target found in the middle of the array
    assert binary_search([1, 2, 3, 4, 5], 3) == 2
    
    # Test case 2: Target found at the beginning of the array
    assert binary_search([1, 2, 3, 4, 5], 1) == 0
    
    # Test case 3: Target found at the end of the array
    assert binary_search([1, 2, 3, 4, 5], 5) == 4
    
    # Test case 4: Target not found in the array
    assert binary_search([1, 2, 3, 4, 5], 6) == -1
    
    # Test case 5: Target found in an array with duplicate elements
    assert binary_search([1, 2, 2, 3, 4, 5], 2) == 1
    
    # Test case 6: Empty array
    with pytest.raises(IndexError):
        binary_search([], 1)
    
    # Test case 7: Array with a single element
    assert binary_search([1], 1) == 0
    
    # Test case 8: Target is less than the smallest element in the array
    with pytest.raises(IndexError):
        binary_search([1, 2, 3, 4, 5], 0)
    
    # Test case 9: Target is greater than the largest element in the array
    with pytest.raises(IndexError):
        binary_search([1, 2, 3, 4, 5], 6)

def test_longest_unique_substring():
    assert longest_unique_substring("abcabcbb") == 3
    assert longest_unique_substring("bbbbbb") == 1
    assert longest_unique_substring("pwwkew") == 3
    assert longest_unique_substring("") == 0
    assert longest_unique_substring("a") == 1
    with pytest.raises(TypeError):
        longest_unique_substring(123)
    with pytest.raises(TypeError):
        longest_unique_substring("abcabcbb")

def test_merge_intervals():
    assert merge_intervals([]) == []
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert merge_intervals([[1, 4], [4, 5]]) == [[1, 5]]
    assert merge_intervals([[1, 3], [5, 8], [4, 10], [20, 25]]) == [[1, 3], [4, 10], [20, 25]]
    with pytest.raises(TypeError):
        merge_intervals([1, 2, 3])
    with pytest.raises(TypeError):
        merge_intervals([[1, 2], 3, 4])
    with pytest.raises(TypeError):
        merge_intervals([[1, 2], [3, 4], '5', 6])

def test_two_sum():
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert two_sum([3, 2, 4], 6) == [1, 0]
    assert two_sum([3, 3], 6) == [0, 1]
    with pytest.raises(TypeError):
        two_sum("123", 9)
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], "9")
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], 9.5)
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], None)
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], True)
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], False)
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], [])
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], {})
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], ())
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], set())
    with pytest.raises(TypeError):
        two_sum([2, 7, 11, 15], 9, 10)
