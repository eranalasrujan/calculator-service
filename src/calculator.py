import math
import itertools
from collections import defaultdict, deque, Counter
from typing import List, Dict, Tuple, Set


def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def prime_factors(n: int) -> List[int]:
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // gcd(a, b)


def flatten_list(data):
    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def longest_common_subsequence(a: str, b: str) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[-1][-1]


def edit_distance(a: str, b: str) -> int:
    dp = [[0]*(len(b)+1) for _ in range(len(a)+1)]
    for i in range(len(a)+1):
        dp[i][0] = i
    for j in range(len(b)+1):
        dp[0][j] = j

    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    return dp[-1][-1]


def knapsack(weights: List[int], values: List[int], capacity: int) -> int:
    n = len(weights)
    dp = [[0]*(capacity+1) for _ in range(n+1)]

    for i in range(1, n+1):
        for w in range(capacity+1):
            if weights[i-1] <= w:
                dp[i][w] = max(
                    dp[i-1][w],
                    values[i-1] + dp[i-1][w-weights[i-1]]
                )
            else:
                dp[i][w] = dp[i-1][w]

    return dp[n][capacity]


def permutations(lst: List[int]) -> List[Tuple[int]]:
    return list(itertools.permutations(lst))


def combinations(lst: List[int], r: int):
    return list(itertools.combinations(lst, r))


def binary_search(arr: List[int], target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


def merge_intervals(intervals: List[List[int]]) -> List[List[int]]:
    intervals.sort()
    merged = []

    for interval in intervals:
        if not merged or merged[-1][1] < interval[0]:
            merged.append(interval)
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])

    return merged


def top_k_frequent(nums: List[int], k: int) -> List[int]:
    count = Counter(nums)
    return [num for num, _ in count.most_common(k)]


def group_anagrams(words: List[str]) -> List[List[str]]:
    groups = defaultdict(list)
    for word in words:
        key = tuple(sorted(word))
        groups[key].append(word)
    return list(groups.values())


def palindrome_partitions(s: str) -> List[List[str]]:
    def is_pal(x):
        return x == x[::-1]

    result = []

    def backtrack(start, path):
        if start == len(s):
            result.append(path[:])
            return
        for i in range(start+1, len(s)+1):
            sub = s[start:i]
            if is_pal(sub):
                backtrack(i, path+[sub])

    backtrack(0, [])
    return result


def matrix_multiply(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
    if not a or not b:
        return []

    rows_a = len(a)
    cols_a = len(a[0])
    rows_b = len(b)
    cols_b = len(b[0])

    if cols_a != rows_b:
        raise ValueError("matrix dimensions do not match for multiplication")

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for k in range(cols_a):
            if a[i][k] == 0:
                continue
            for j in range(cols_b):
                result[i][j] += a[i][k] * b[k][j]

    return result


def transpose_matrix(matrix: List[List[int]]) -> List[List[int]]:
    return list(map(list, zip(*matrix)))


def shortest_path_bfs(graph: Dict[int, List[int]], start: int, end: int) -> int:
    queue = deque([(start, 0)])
    visited = {start}

    while queue:
        node, dist = queue.popleft()
        if node == end:
            return dist
        for neigh in graph.get(node, []):
            if neigh not in visited:
                visited.add(neigh)
                queue.append((neigh, dist+1))
    return -1


def detect_cycle_directed(graph: Dict[int, List[int]]) -> bool:
    visited = set()
    stack = set()

    def dfs(node):
        if node in stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        stack.add(node)

        for neigh in graph.get(node, []):
            if dfs(neigh):
                return True

        stack.remove(node)
        return False

    return any(dfs(n) for n in graph)


def dijkstra(graph: Dict[int, List[Tuple[int,int]]], start: int):
    import heapq
    dist = {node: math.inf for node in graph}
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue

        for neigh, w in graph[node]:
            nd = d + w
            if nd < dist.get(neigh, math.inf):
                dist[neigh] = nd
                heapq.heappush(pq, (nd, neigh))

    return dist


def sliding_window_max(nums: List[int], k: int) -> List[int]:

    if k <= 0:
        raise ValueError("k must be positivee")

    if not nums:
        return []

    if k > len(nums):
        return [max(nums)]

    dq = deque()
    result = []

    for i, n in enumerate(nums):

        while dq and dq[0] <= i-k:
            dq.popleft()

        while dq and nums[dq[-1]] < n:
            dq.pop()

        dq.append(i)

        if i >= k-1:
            result.append(nums[dq[0]])

    return result


def majority_element(nums: List[int]) -> int:
    count = 0
    candidate = None

    for n in nums:
        if count == 0:
            candidate = n
        count += 1 if n == candidate else -1

    return candidate


def roman_to_int(s: str) -> int:
    vals = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}
    total = 0
    prev = 0

    for c in reversed(s):
        val = vals[c]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val

    return total


def int_to_roman(num: int) -> str:
    vals = [
        (1000,"M"),(900,"CM"),(500,"D"),(400,"CD"),
        (100,"C"),(90,"XC"),(50,"L"),(40,"XL"),
        (10,"X"),(9,"IX"),(5,"V"),(4,"IV"),(1,"I")
    ]

    result = ""
    for v, sym in vals:
        while num >= v:
            result += sym
            num -= v

    return result


def evaluate_rpn(tokens: List[str]) -> int:
    stack = []

    for t in tokens:
        if t in {"+","-","*","/"}:
            b = stack.pop()
            a = stack.pop()
            if t == "+":
                stack.append(a+b)
            elif t == "-":
                stack.append(a-b)
            elif t == "*":
                stack.append(a*b)
            else:
                stack.append(int(a/b))
        else:
            stack.append(int(t))

    return stack[0]


def generate_subsets(nums: List[int]) -> List[List[int]]:
    result = [[]]

    for n in nums:
        result += [curr+[n] for curr in result]

    return result


def word_frequency(text: str) -> Dict[str,int]:
    words = text.lower().split()
    return dict(Counter(words))


def longest_unique_substring(s: str) -> int:
    seen = {}
    left = 0
    best = 0

    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1

        seen[ch] = right
        best = max(best, right-left+1)

    return best