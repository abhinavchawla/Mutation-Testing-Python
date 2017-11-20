import collections
INT_MAX = 4294967296
INT_MIN = -4294967296
# def swap(arr,i, j):
#         arr[i], arr[j] = arr[j], arr[i]
#
# def bubble_sort(arr):
#     n = len(arr)
#     swapped = True
#     while swapped:
#         swapped = False
#         for i in range(1, n):
#             if arr[i - 1] > arr[i]:
#                 swap(arr,i - 1, i)
#                 swapped = True

# def comb_sort(arr):
#     n = len(arr)
#     gap = n
#     shrink = 1.3
#     sorted = False
#     while not sorted:
#         gap = int(floor(gap/shrink))
#         if gap > 1:
#             sorted = False
#         else:
#             gap = 1
#             sorted = True
#
#         i = 0
#         while i + gap < n:
#             if arr[i] > arr[i + gap]:
#                 swap(arr,i, i + gap)
#                 sorted = False
#             i = i + 1
#
#
# def counting_sort(arr):
# 	"""
#     Counting_sort
# 	Sorting a array which has no element greater than k
# 	Creating a new temp_arr,where temp_arr[i] contain the number of
# 	element less than or equal to i in the arr
#     Then placing the number i into a correct position in the result_arr
# 	return the result_arr
# 	Complexity: 0(n)
# 	"""
#
# 	m = min(arr)
# 	#in case there are negative elements, change the array to all positive element
# 	different = 0
# 	if m < 0:
# 		#save the change, so that we can convert the array back to all positive number
# 		different = -m
# 		for i in range (len(arr)):
# 			arr[i]+= -m
# 	k = max(arr)
# 	temp_arr = [0]*(k+1)
# 	for i in range(0,len(arr)):
# 		temp_arr[arr[i]] = temp_arr[arr[i]]+1
# 	#temp_array[i] contain the times the number i appear in arr
#
# 	for i in range(1, k+1):
# 		temp_arr[i] = temp_arr[i] + temp_arr[i-1]
# 	#temp_array[i] contain the number of element less than or equal i in arr
#
# 	result_arr = [0]*len(arr)
# 	#creating a result_arr an put the element in a correct positon
# 	for i in range(len(arr)-1,-1,-1):
# 		result_arr[temp_arr[arr[i]]-1] = arr[i]-different
# 		temp_arr[arr[i]] = temp_arr[arr[i]]-1
#
# 	return result_arr
#
# def max_heap_sort(arr):
#     """ Heap Sort that uses a max heap to sort an array in ascending order
#         Complexity: O(n log(n))
#     """
#     for i in range(len(arr)-1,0,-1):
#         max_heapify(arr, i)
#
#         temp = arr[0]
#         arr[0] = arr[i]
#         arr[i] = temp
#
#
# def max_heapify(arr, end):
#     """ Max heapify helper for max_heap_sort
#     """
#     last_parent = int((end-1)/2)
#
#     # Iterate from last parent to first
#     for parent in range(last_parent,-1,-1):
#         current_parent = parent
#
#         # Iterate from current_parent to last_parent
#         while current_parent <= last_parent:
#             # Find greatest child of current_parent
#             child = 2*current_parent + 1
#             if child + 1 <= end and arr[child] < arr[child+1]:
#                 child = child + 1
#
#             # Swap if child is greater than parent
#             if arr[child] > arr[current_parent]:
#                 temp = arr[current_parent]
#                 arr[current_parent] = arr[child]
#                 arr[child] = temp
#
#                 current_parent = child
#             # If no swap occured, no need to keep iterating
#             else:
#                 break
#
#
# def min_heap_sort(arr):
#     """ Heap Sort that uses a min heap to sort an array in ascending order
#         Complexity: O(n log(n))
#     """
#     for i in range(0, len(arr)-1):
#         min_heapify(arr, i)
#
#
# def min_heapify(arr, start):
#     """ Min heapify helper for min_heap_sort
#     """
#     # Offset last_parent by the start (last_parent calculated as if start index was 0)
#     # All array accesses need to be offet by start
#     end = len(arr)-1
#     last_parent = int((end-start-1)/2)
#
#     # Iterate from last parent to first
#     for parent in range(last_parent,-1,-1):
#         current_parent = parent
#
#         # Iterate from current_parent to last_parent
#         while current_parent <= last_parent:
#             # Find lesser child of current_parent
#             child = 2*current_parent + 1
#             if child + 1 <= end-start and arr[child+start] > arr[child+1+start]:
#                 child = child + 1
#
#             # Swap if child is less than parent
#             if arr[child+start] < arr[current_parent+start]:
#                 temp = arr[current_parent+start]
#                 arr[current_parent+start] = arr[child+start]
#                 arr[child+start] = temp
#
#                 current_parent = child
#             # If no swap occured, no need to keep iterating
#             else:
#                 break
#
# def insertion_sort(arr):
#     """ Insertion Sort
#         Complexity: O(n^2)
#     """
#     for i in xrange(len(arr)):
#         cursor = arr[i]
#         pos = i
#         while pos > 0 and arr[pos-1] > cursor:
#             # Swap the number down the list
#             arr[pos] = arr[pos-1]
#             pos = pos-1
#         # Break and do the final swap
#         arr[pos] = cursor
#     return arr
#
# def can_attend_meetings(intervals):
#     """
#     :type intervals: List[Interval]
#     :rtype: bool
#     """
#     intervals = sorted(intervals, key=lambda x: x.start)
#     for i in range(1, len(intervals)):
#         if intervals[i].start < intervals[i-1].end:
#             return False
#     return True
#
# def merge_sort(arr):
#     """ Merge Sort
#         Complexity: O(n log(n))
#     """
#     # Our recursive base case
#     if len(arr)<= 1:
#         return arr
#     mid = len(arr)/2
#     # Perform merge_sort recursively on both halves
#     left, right = merge_sort(arr[mid:]), merge_sort(arr[:mid])
#
#     # Merge each side together
#     return merge(left, right)
#
# def merge(left, right):
#     """ Merge helper
#         Complexity: O(n)
#     """
#     arr = []
#     left_cursor, right_cursor = 0,0
#     while left_cursor < len(left) and right_cursor < len(right):
#         # Sort each one and place into the result
#         if left[left_cursor] <= right[right_cursor]:
#             arr.append(left[left_cursor])
#             left_cursor+=1
#         else:
#             arr.append(right[right_cursor])
#             right_cursor+=1
#    # Add the left overs if there's any left to the result
#     for i in range(left_cursor,len(left)):
#         arr.append(left[i])
#     for i in range(right_cursor,len(right)):
#         arr.append(right[i])
#
#    # Return result
#     return arr
#
# def quick_sort(arr, first, last):
#     """ Quicksort
#         Complexity: best O(n) avg O(n log(n)), worst O(N^2)
#     """
#     if first < last:
#         pos = partition(arr, first, last)
#         print(arr[first:pos-1], arr[pos+1:last])
#         # Start our two recursive calls
#         quick_sort(arr, first, pos-1)
#         quick_sort(arr, pos+1, last)
#
# def partition(arr, first, last):
#     wall = first
#     for pos in range(first, last):
#         if arr[pos] < arr[last]: # last is the pivot
#             arr[pos], arr[wall] = arr[wall], arr[pos]
#             wall += 1
#     arr[wall], arr[last] = arr[last], arr[wall]
#     print(wall)
#     return wall
#
# def selection_sort(arr):
#     """ Selection Sort
#         Complexity: O(n^2)
#     """
#     for i in xrange(len(arr)):
#         minimum = i
#         for j in xrange(i+1, len(arr)):
#             # "Select" the correct value
#             if arr[j] < arr[minimum]:
#                 minimum = j
#         # Using a pythonic swap
#         arr[minimum], arr[i] = arr[i], arr[minimum]
#     return arr
#
# def sort_colors(nums):
#     i = j = 0
#     for k in range(len(nums)):
#         v = nums[k]
#         nums[k] = 2
#         if v < 2:
#             nums[j] = 1
#             j += 1
#         if v == 0:
#             nums[i] = 0
#             i += 1
#
# def wiggle_sort(nums):
#     for i in range(len(nums)):
#         if (i % 2 == 1) == (nums[i-1] > nums[i]):
#             nums[i-1], nums[i] = nums[i], nums[i-1]
#     return nums
#

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

    def __eq__(self, other):
            return self.val == other.val and self.left == other.left and self.right == other.right
# """
# Given two binary trees, write a function to check
# if they are equal or not.
#
# Two binary trees are considered equal if they are
# structurally identical and the nodes have the same value.
# """
def isSameTree(p, q):
    if not p and not q:
        return True
    if p and q and p.val == q.val:
        return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
    return False

# """
# Given a binary tree and a sum, find all root-to-leaf
# paths where each path's sum equals the given sum.
#
# For example:
# Given the below binary tree and sum = 22,
#               5
#              / \
#             4   8
#            /   / \
#           11  13  4
#          /  \    / \
#         7    2  5   1
# return
# [
#    [5,4,11,2],
#    [5,8,4,5]
# ]
# """
def path_sum(root, sum):
    if not root:
        return []
    res = []
    DFS(root, sum, [], res)
    return res

def DFS(root, sum, ls, res):
    if not root.left and not root.right and root.val == sum:
        ls.append(root.val)
        res.append(ls)
    if root.left:
        DFS(root.left, sum-root.val, ls+[root.val], res)
    if root.right:
        DFS(root.right, sum-root.val, ls+[root.val], res)

#
# DFS with stack
def path_sum2(root, s):
    if not root:
        return []
    res = []
    stack = [(root, [root.val])]
    while stack:
        node, ls = stack.pop()
        if not node.left and not node.right and sum(ls) == s:
            res.append(ls)
        if node.left:
            stack.append((node.left, ls+[node.left.val]))
        if node.right:
            stack.append((node.right, ls+[node.right.val]))
    return res


#BFS with queue
def path_sum3(root, sum):
    if not root:
        return []
    res = []
    queue = [(root, root.val, [root.val])]
    while queue:
        node, val, ls = queue.pop(0)  # popleft
        if not node.left and not node.right and val == sum:
            res.append(ls)
        if node.left:
            queue.append((node.left, val+node.left.val, ls+[node.left.val]))
        if node.right:
            queue.append((node.right, val+node.right.val, ls+[node.right.val]))
    return res
# """
# Given a binary tree and a sum, determine if the tree has a root-to-leaf
# path such that adding up all the values along the path equals the given sum.
#
# For example:
# Given the below binary tree and sum = 22,
#               5
#              / \
#             4   8
#            /   / \
#           11  13  4
#          /  \      \
#         7    2      1
# return true, as there exist a root-to-leaf path 5->4->11->2 which sum is 22.
# """
def has_path_sum(root, sum):
    # """
    # :type root: TreeNode
    # :type sum: int
    # :rtype: bool
    # """
    if not root:
        return False
    if not root.left and not root.right and root.val == sum:
        return True
    sum -= root.val
    return has_path_sum(root.left, sum) or has_path_sum(root.right, sum)


# DFS with stack
def has_path_sum2(root, sum):
    if not root:
        return False
    stack = [(root, root.val)]
    while stack:
        node, val = stack.pop()
        if not node.left and not node.right:
            if val == sum:
                return True
        if node.left:
            stack.append((node.left, val+node.left.val))
        if node.right:
            stack.append((node.right, val+node.right.val))
    return False


# BFS with queue
def has_path_sum3(root, sum):
    if not root:
        return False
    queue = [(root, sum-root.val)]
    while queue:
        node, val = queue.pop(0)  # popleft
        if not node.left and not node.right:
            if val == 0:
                return True
        if node.left:
            queue.append((node.left, val-node.left.val))
        if node.right:
            queue.append((node.right, val-node.right.val))
    return False

# """
# Given a binary tree, find its maximum depth.
#
# The maximum depth is the number of nodes along the
# longest path from the root node down to the farthest leaf node.
# """

def max_height(root):
    if not root:
        return 0
    height = 0
    queue = [root]
    while queue:
        height += 1
        level = []
        while queue:
            node = queue.pop(0)
            if node.left:
                level.append(node.left)
            if node.right:
                level.append(node.right)
        queue = level
    return height
# """
# Given a binary tree, find the lowest common ancestor
# (LCA) of two given nodes in the tree.
#
# According to the definition of LCA on Wikipedia:
#     “The lowest common ancestor is defined between two nodes
#     v and w as the lowest node in T that has both v and w as
#     descendants
#     (where we allow a node to be a descendant of itself).”
#
#         _______3______
#        /              \
#     ___5__          ___1__
#    /      \        /      \
#    6      _2       0       8
#          /  \
#          7   4
# For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3.
# Another example is LCA of nodes 5 and 4 is 5,
# since a node can be a descendant of itself according to the LCA definition.
# """
def LCA(root, p, q):
    # """
    # :type root: TreeNode
    # :type p: TreeNode
    # :type q: TreeNode
    # :rtype: TreeNode
    # """
    if not root or root is p or root is q:
        return root
    left = LCA(root.left, p, q)
    right = LCA(root.right, p, q)
    if left and right:
        return root
    return left if left else right

# """
# Given a binary tree, check whether it is a mirror of
# itself (ie, symmetric around its center).
#
# For example, this binary tree [1,2,2,3,4,4,3] is symmetric:
#
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
# But the following [1,2,2,null,3,null,3] is not:
#     1
#    / \
#   2   2
#    \   \
#    3    3
#  """
def is_symmetric(root):
    if not root:
        return True
    return helper(root.left, root.right)


def helper(p, q):
    if not p and not q:
        return True
    if not p or not q or q.val != p.val:
        return False
    return helper(p.left, q.right) and helper(p.right, q.left)


def is_symmetric_iterative(root):
    if not root:
        return True
    stack = [[root.left, root.right]]
    while stack:
        left, right = stack.pop()  # popleft
        if not left and not right:
            continue
        if not left or not right:
            return False
        if left.val == right.val:
            stack.append([left.left, right.right])
            stack.append([left.right, right.right])
        else:
            return False
    return True

# Given two binary trees s and t, check if t is a subtree of s.
# A subtree of a tree t is a tree consisting of a node in t and
# all of its descendants in t.

# Example 1:

# Given s:

     # 3
    # / \
   # 4   5
  # / \
 # 1   2

# Given t:

   # 4
  # / \
 # 1   2
# Return true, because t is a subtree of s.

def is_subtree(big, small):
    flag = False
    queue = collections.deque()
    queue.append(big)
    while queue:
        node = queue.popleft()
        if node.val == small.val:
            flag = comp(node, small)
            break
        else:
            queue.append(node.left)
            queue.append(node.right)
    return flag

def comp(p, q):
    if not p and not q:
        return True
    if p and q:
        return p.val == q.val and comp(p.left,q.left) and comp(p.right, q.right)
    return False

# """
# Given a binary tree, check whether it is balanced (A tree where no leaf is much farther away from the root than any other leaf) .
# An empty tree is height-balanced. A non-empty binary tree T is balanced if:
# 1) Left subtree of T is balanced
# 2) Right subtree of T is balanced
# 3) The difference between heights of left subtree and right subtree is not more than 1.
#
# For example, this binary tree [1,2,2,3,4,4,3] is balanced:
#
#     1
#    / \
#   2   2
#  / \ / \
# 3  4 4  3
#
#  """
def is_balanced(root):
    # """
    # O(N) solution
    # """
    return -1 != get_depth(root)

def get_depth(root):
    # """
    # return 0 if unbalanced else depth + 1
    # """
    if not root:
        return 0
    left  = get_depth(root.left)
    right = get_depth(root.right)
    if abs(left-right) > 1:
        return -1
    return 1 + max(left, right)

################################

def is_balanced_2(root):
    # """
    # O(N^2) solution
    # """
    left = max_height(root.left)
    right = max_height(root.right)
    if not left and not right:
        return True
    if not left or not right:
        return False
    return abs(left-right) <= 1 and is_balanced_2(root.left) and is_balanced_2(root.right)

def reverse(root):
    if not root:
        return
    root.left, root.right = root.right, root.left
    if root.left:
        reverse(root.left)
    if root.right:
        reverse(root.right)

# Given a binary tree, find the deepest node
# that is the left child of its parent node.

# Example:

     # 1
   # /   \
  # 2     3
 # / \     \
# 4   5     6
           # \
            # 7
# should return 4.
class DeepestLeft:
    def __init__(self):
        self.depth = 0
        self.Node = None

def find_deepest_left(root, is_left, depth, res):
    if not root:
        return
    if is_left and depth > res.depth:
        res.depth = depth
        res.Node = root
    find_deepest_left(root.left, True, depth + 1, res)
    find_deepest_left(root.right, False, depth + 1, res)

def bintree2list(root):
    # """
    # type root: root class
    # """
    if not root:
        return root
    root = bintree2list_util(root)
    while root.left:
        root = root.left
    return return_tree_list(root)

def bintree2list_util(root):
    if not root:
        return root
    if root.left:
        left = bintree2list_util(root.left)
        while left.right:
            left = left.right
        left.right = root
        root.left = left
    if root.right:
        right = bintree2list_util(root.right)
        while right.left:
            right = right.left
        right.left = root
        root.right = right
    return root

def return_tree_list(root):
    lst=[]
    while root:
        lst.append(root.val)
        root = root.right
    return lst
# """
# Given a binary tree, return the inorder traversal of
# its nodes' values. (ie, left subtree, then the root, and then the right subtree).
#
# For example:
# Given binary tree [3,9,20,null,null,15,7],
#     3
#    / \
#   9  20
#     /  \
#    15   7
# return its inorder traversal as: [9,3,15,20,7]
# """
def inorder(root):
    res = []
    if not root:
        return res
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        res.append(root.val)
        root = root.right
    return res

# """
# Given a binary tree, return the level order traversal of
# its nodes' values. (ie, from left to right, level by level).
#
# For example:
# Given binary tree [3,9,20,null,null,15,7],
#     3
#    / \
#   9  20
#     /  \
#    15   7
# return its level order traversal as:
# [
#   [3],
#   [9,20],
#   [15,7]
# ]
# """

def level_order(root):
    ans = []
    if not root:
        return ans
    level = [root]
    while level:
        current = []
        new_level = []
        for node in level:
            current.append(node.val)
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level = new_level
        ans.append(current)
    return ans
# """
# Given a binary tree, return the zigzag level order traversal
# of its nodes' values.
# (ie, from left to right, then right to left
# for the next level and alternate between).
#
# For example:
# Given binary tree [3,9,20,null,null,15,7],
#     3
#    / \
#   9  20
#     /  \
#    15   7
# return its zigzag level order traversal as:
# [
#   [3],
#   [20,9],
#   [15,7]
# ]
# """
def zigzag_level(root):
    res = []
    if not root:
        return res
    level = [root]
    flag = 1
    while level:
        current = []
        new_level = []
        for node in level:
            current.append(node.val)
            if node.left:
                new_level.append(node.left)
            if node.right:
                new_level.append(node.right)
        level = new_level
        res.append(current[::flag])
        flag *= -1
    return res
# """
# Given an array where elements are sorted in ascending order,
# convert it to a height balanced BST.
# """
def array2bst(nums):
    if not nums:
        return None
    mid = len(nums)//2
    node = TreeNode(nums[mid])
    node.left = array2bst(nums[:mid])
    node.right = array2bst(nums[mid+1:])
    return node
# Given a non-empty binary search tree and a target value,
# find the value in the BST that is closest to the target.

# Note:
# Given target value is a floating point.
# You are guaranteed to have only one unique value in the BST
# that is closest to the target.


# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
def closest_value(root, target):
    # """
    # :type root: TreeNode
    # :type target: float
    # :rtype: int
    # """
    a = root.val
    kid = root.left if target < a else root.right
    if not kid:
        return a
    b = closest_value(kid, target)
    return min((a,b), key=lambda x: abs(target-x))

# """
# Given a binary tree, determine if it is a valid binary search tree (BST).
#
# Assume a BST is defined as follows:
#
# The left subtree of a node contains only nodes
# with keys less than the node's key.
# The right subtree of a node contains only nodes
# with keys greater than the node's key.
# Both the left and right subtrees must also be binary search trees.
# Example 1:
#     2
#    / \
#   1   3
# Binary tree [2,1,3], return true.
# Example 2:
#     1
#    / \
#   2   3
# Binary tree [1,2,3], return false.
# """

def isBST(node):
    return (isBSTUtil(node, INT_MIN, INT_MAX))

def isBSTUtil(node, mini, maxi):

    if node is None:
        return True

    if node.val < mini or node.val > maxi:
        return False

    return (isBSTUtil(node.left, mini, node.val -1) and
          isBSTUtil(node.right, node.val+1, maxi))

def num_trees(n):
    # """
    # :type n: int
    # :rtype: int
    # """
    dp = [0] * (n+1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n+1):
        for j in range(i+1):
            dp[i] += dp[i-j] * dp[j-1]
    return dp[-1]

def successor(root, node):
    succ = None
    while root:
        if node.val < root.val:
            succ = root
            root = root.left
        else:
            root = root.right
    return succ
