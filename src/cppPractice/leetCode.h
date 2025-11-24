// this file will contain all datastructures needed in leetCode.cpp

#include <iostream>
#include <vector>
#include <string>
using namespace std;

struct TreeNode {
  int val;
  TreeNode* left;
  TreeNode* right;
  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode* left, TreeNode* right)
      : val(x), left(left), right(right) {}
};

class Solution {
 private:
  // any private helper functions will be here

  // takes in a vector reference and returns true/false depending on if we have
  // a valid target sum
  bool twoSumBool(vector<int>& vect, int target);

  //this will take in a vector, and current index
  //and will smoothen 
  int smoothen(vector<vector<int>> &mat, int x, int y);

  //this method will traverse a tree, takes in a value representing root for comparisons
  int dfsVal(TreeNode * root, int rootVal);

 public:
  Solution(/* args */) = default;
  ~Solution() = default;

  // below are the functions to implement:

  // Given the root of a binary search tree and an integer k,
  // return true if there exist two elements in the BST such that their sum is
  // equal to k, or false otherwise.
  bool findTarget(TreeNode* root, int k);

  // We have two special characters:
  // The first character can be represented by one bit 0.
  // The second character can be represented by two bits (10 or 11).
  // Given a binary array bits that ends with 0, return true if the last
  // character must be a one-bit character.
  bool isOneBitCharacter(vector<int>& bits);

  // There is a robot starting at the position (0, 0), the origin, on a 2D
  // plane. Given a sequence of its moves, judge
  // if this robot ends up at (0, 0) after it completes its moves.
  // You are given a string moves that represents the move sequence of the robot
  // where moves[i] represents
  // its ith move. Valid moves are 'R' (right), 'L' (left), 'U' (up), and 'D'
  // (down).
  // Return true if the robot returns to the origin after it finishes all of its
  // moves, or false otherwise. Note: The way that the robot is "facing" is
  // irrelevant. 'R' will always make the robot move
  // to the right once, 'L' will always make it move left, etc. Also, assume
  // that the magnitude of the robot's movement is the same for each move.
  bool judgeCircle(string moves);

  // An image smoother is a filter of the size 3 x 3 that can be applied to each cell of an image by 
  // rounding down the average of the cell and the eight surrounding cells (i.e., the average of the 
  //   nine cells in the blue smoother). If one or more of the surrounding cells of a cell is not present, 
  //   we do not consider it in the average (i.e., the average of the four cells in the red smoother).
  vector<vector<int>> imageSmoother(vector<vector<int>>& img);

  // You are given an array of integers nums. You are also given an integer original which is the first number that needs to be searched for in nums.
  // You then do the following steps:
  // If original is found in nums, multiply it by two (i.e., set original = 2 * original).
  // Otherwise, stop the process.
  // Repeat this process with the new number as long as you keep finding the number.
  // Return the final value of original.
  int findFinalValue(vector<int>& nums, int original);

  //given root of binary tree, find the second smallest node,
  //each node is the min of left and right subtrees
  int findSecondMinimumValue(TreeNode* root);

  // Given an unsorted array of integers nums, return the length of the longest continuous 
  // increasing subsequence (i.e. subarray). The subsequence must be strictly increasing.
  // A continuous increasing subsequence is defined by two indices l and r (l < r) such that 
  // it is [nums[l], nums[l + 1], ..., nums[r - 1], nums[r]] and for each l <= i < r, nums[i] < nums[i + 1].
  int findLengthOfLCIS(vector<int>& nums);

  //Given a string s, return true if the s can be palindrome after deleting at most one character from it.
  bool validPalindrome(string s);
};