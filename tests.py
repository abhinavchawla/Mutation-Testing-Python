from unittest import TestCase
import source


def createNode():
    a1 = source.TreeNode(5)
    a2 = source.TreeNode(4)
    a3 = source.TreeNode(8)
    a4 = source.TreeNode(11)
    a5 = source.TreeNode(13)
    a6 = source.TreeNode(4)
    a7 = source.TreeNode(7)
    a8 = source.TreeNode(2)
    a9 = source.TreeNode(5)
    a10 = source.TreeNode(1)
    a1.left = a2
    a1.right =a3
    a2.left = a4
    a3.left = a5
    a3.right = a6
    a4.left = a7
    a4.right = a8
    a6.left = a9
    a6.right = a10
    return a1

def createBSTNode():
    a1 = source.TreeNode(8)
    a2 = source.TreeNode(7)
    a3 = source.TreeNode(10)
    a4 = source.TreeNode(4)
    a5 = source.TreeNode(9)
    a6 = source.TreeNode(15)
    a7 = source.TreeNode(3)
    a8 = source.TreeNode(6)
    a9 = source.TreeNode(13)
    a10 = source.TreeNode(16)
    a1.left = a2
    a1.right =a3
    a2.left = a4
    a3.left = a5
    a3.right = a6
    a4.left = a7
    a4.right = a8
    a6.left = a9
    a6.right = a10
    return a1


class CalculatorTest(TestCase):

    def test_isSameTree(self):
        a1 = source.TreeNode(30)
        a2 = source.TreeNode(40)
        a3 = source.TreeNode(50)
        a1.left = a2
        a1.right = a3
        b1 = source.TreeNode(30)
        b2 = source.TreeNode(40)
        b3 = source.TreeNode(50)
        b1.left = b2
        b1.right = b3
        self.assertEqual(source.isSameTree(a1,b1),True)

    def test_isSameTree_False(self):
        a1 = source.TreeNode(30)
        a2 = source.TreeNode(40)
        a3 = source.TreeNode(50)
        a1.left = a2
        a1.right = a3
        b1 = source.TreeNode(30)
        b2 = source.TreeNode(90)
        b3 = source.TreeNode(50)
        b1.left = b2
        b1.right = b3
        self.assertEqual(source.isSameTree(a1,b1),False)


    def test_pathSum(self):
        a1 = createNode()
        self.assertEqual(source.path_sum(a1,22),[[5,4,11,2],[5,8,4,5]] )


    def test_pathSum2(self):
        a1 = createNode()
        self.assertEqual(source.path_sum2(a1,22),[[5,8,4,5],[5,4,11,2]] )

    def test_pathSum3(self):
        a1 = createNode()
        self.assertEqual(source.path_sum3(a1,22),[[5,4,11,2],[5,8,4,5]] )
    #
    def test_pathSum4(self):
        a1 = createNode()
        print(source.has_path_sum(a1,22))
        self.assertEqual(source.has_path_sum(a1,22),True)

    def test_pathSum5(self):
        a1 = createNode()
        a1.val =6
        self.assertEqual(source.has_path_sum(a1,22),False)

    def test_pathSum6(self):
        a1 = createNode()
        print(source.has_path_sum(a1,22))
        self.assertEqual(source.has_path_sum2(a1,22),True)

    def test_pathSum7(self):
        a1 = createNode()
        a1.val =6
        self.assertEqual(source.has_path_sum2(a1,22),False)

    def test_pathSum8(self):
        a1 = createNode()
        print(source.has_path_sum(a1,22))
        self.assertEqual(source.has_path_sum3(a1,22),True)

    def test_pathSum9(self):
        a1 = createNode()
        a1.val =6
        self.assertEqual(source.has_path_sum3(a1,22),False)

    def test_maxheight(self):
        a1 = createNode()
        self.assertEqual(source.max_height(a1),4)

    def test_LCA(self):
        a1 = source.TreeNode(5)
        a2 = source.TreeNode(4)
        a3 = source.TreeNode(8)
        a4 = source.TreeNode(11)
        a5 = source.TreeNode(13)
        a6 = source.TreeNode(4)
        a7 = source.TreeNode(7)
        a8 = source.TreeNode(2)
        a9 = source.TreeNode(5)
        a10 = source.TreeNode(1)
        a1.left = a2
        a1.right =a3
        a2.left = a4
        a3.left = a5
        a3.right = a6
        a4.left = a7
        a4.right = a8
        a6.left = a9
        a6.right = a10
        self.assertEqual(source.LCA(a1,a5,a10),a3)

    def test_issymmetric(self):
        a1=createNode()
        self.assertEqual(source.is_symmetric(a1),False)

    def test_issymmetric2(self):
        a1 = source.TreeNode(1)
        a2 = source.TreeNode(2)
        a3 = source.TreeNode(2)
        a4 = source.TreeNode(3)
        a5 = source.TreeNode(4)
        a6 = source.TreeNode(4)
        a7 = source.TreeNode(3)
        a1.left = a2
        a1.right =a3
        a2.left = a4
        a2.right = a5
        a3.left = a6
        a3.right = a7
        self.assertEqual(source.is_symmetric(a1),True)
    def test_issymmetric3(self):
        a1=createNode()
        self.assertEqual(source.is_symmetric_iterative(a1),False)

    def test_issymmetric4(self):
        a1 = source.TreeNode(1)
        a2 = source.TreeNode(2)
        a3 = source.TreeNode(2)
        a1.left = a2
        a1.right =a3
        self.assertEqual(source.is_symmetric_iterative(a1),True)

    def test_issubtree(self):
        a1 = source.TreeNode(11)
        a2 = source.TreeNode(7)
        a3 = source.TreeNode(2)
        a1.left = a2
        a1.right =a3
        b1 = createNode()
        self.assertEqual(source.is_subtree(b1,a1),True)

    def test_issubtree2(self):
        a1 = source.TreeNode(4)
        a2 = source.TreeNode(11)
        a3 = source.TreeNode(1)
        a1.left = a2
        a1.right =a3
        b1 = createNode()
        self.assertEqual(source.is_subtree(b1,a1),False)

    def test_isbalanced(self):
        a1 = createNode()
        self.assertEqual(source.is_balanced(a1),False)
    def test_isbalanced2(self):
        a1 = source.TreeNode(1)
        a2 = source.TreeNode(2)
        a3 = source.TreeNode(2)
        a4 = source.TreeNode(3)
        a5 = source.TreeNode(4)
        a6 = source.TreeNode(4)
        a7 = source.TreeNode(3)
        a1.left = a2
        a1.right =a3
        a2.left = a4
        a2.right = a5
        a3.left = a6
        a3.right = a7
        self.assertEqual(source.is_balanced(a1),True)

    def test_isbalanced3(self):
        a1 = createNode()
        self.assertEqual(source.is_balanced_2(a1),False)
    def test_isbalanced4(self):
        a1 = source.TreeNode(1)
        a2 = source.TreeNode(2)
        a3 = source.TreeNode(2)
        a4 = source.TreeNode(3)
        a5 = source.TreeNode(4)
        a6 = source.TreeNode(4)
        a7 = source.TreeNode(3)
        a1.left = a2
        a1.right =a3
        a2.left = a4
        a2.right = a5
        a3.left = a6
        a3.right = a7
        self.assertEqual(source.is_balanced_2(a1),True)

    def test_deepestleft(self):
        a1=createNode()
        d=source.DeepestLeft()
        source.find_deepest_left(a1,True,1,d)
        self.assertEqual(d.Node,a1.left.left.left)
        self.assertEqual(d.depth,4)
    #
    def test_deepestleft2(self):
        root = source.TreeNode(1)
        root.right = source.TreeNode(3)
        root.right.left = source.TreeNode(4)
        root.right.right = source.TreeNode(6)
        root.right.right.right = source.TreeNode(7)
        d=source.DeepestLeft()
        source.find_deepest_left(root,True,1,d)
        self.assertEqual(d.Node,root.right.left)
        self.assertEqual(d.depth,3)

    def test_reverse(self):
        a1 = source.TreeNode(1)
        a2 = source.TreeNode(4)
        a3 = source.TreeNode(5)
        a4 = source.TreeNode(7)
        a5 = source.TreeNode(9)
        a1.left = a2
        a1.right = a3
        a1.left.left = a4
        a3.right = a5
        b1 = source.TreeNode(1)
        b2 = source.TreeNode(5)
        b3 = source.TreeNode(4)
        b4 = source.TreeNode(7)
        b5 = source.TreeNode(9)
        b1.left = b2
        b1.right = b3
        b1.right.right = b4
        b2.left = b5
        source.reverse(a1)
        self.assertEqual(a1,b1)

    def test_bin2list(self):
        a1 = createNode()
        a = source.bintree2list(a1)
        self.assertEqual(a,[7, 11, 2, 4, 5, 13, 8, 5, 4, 1])

    def test_inorder(self):
        a1 = createNode()
        x = source.inorder(a1)
        self.assertEqual(x,[7, 11, 2, 4, 5, 13, 8, 5, 4, 1])

    def test_levelorder(self):
        a1 = createNode()
        x = source.level_order(a1)
        self.assertEqual(x,[[5], [4, 8], [11, 13, 4], [7, 2, 5, 1]])


    def test_zigzagorder(self):
        a1 = createNode()
        x = source.zigzag_level(a1)
        print(x)
        self.assertEqual(a1,a1)
        self.assertEqual(x,[[5], [8, 4], [11, 13, 4], [1, 5, 2, 7]])

    def test_array2bst(self):
        a1 = source.TreeNode(5)
        a2 = source.TreeNode(2)
        a3 = source.TreeNode(7)
        a1.left = a2
        a1.right = a3
        nums = [2,5,7]
        x = source.array2bst(nums)
        self.assertEqual(x,a1)

    def test_closestvalue(self):
        a1 = createBSTNode()
        x=source.closest_value(a1,20)
        print(x)
        self.assertEqual(a1,a1)
        self.assertEqual(x,16)

    def test_isbst(self):
        a1=createBSTNode()
        x = source.isBST(a1)
        self.assertEqual(x,True)

    def test_isbst2(self):
        a1=createNode()
        x = source.isBST(a1)
        self.assertEqual(x,False)
    def test_numtrees(self):
        x= source.num_trees(5)
        print(x)
        self.assertEqual(x,42)

    def test_successor(self):
        a1 = createBSTNode()
        x= source.successor(a1,a1.right)
        self.assertEqual(x,a1.right.right.left)


    # def test_bubbleSort(self):
    #     arr= [4,2,1,3]
    #     source.bubble_sort(arr)
    #     self.assertEqual(arr,[1,2,3,4])
    #
    # def test_bubbleSort2(self):
    #     arr= [7,1,4,2,1,3]
    #     source.bubble_sort(arr)
    #     self.assertEqual(arr,[1,1,2,3,4,7])
