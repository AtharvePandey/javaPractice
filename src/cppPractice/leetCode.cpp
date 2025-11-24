#include "./leetCode.h"

#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>
#include <queue>
#include <string>
#include <unordered_set>
#include <vector>
using namespace std;

bool Solution::twoSumBool(vector<int>& vect, int target) {
  // we don't need to keep indecies, we just need to go through vector
  // and store target - num, if its in set, we return true

  unordered_set<int> set;

  for (auto it = vect.begin(); it != vect.end(); ++it) {
    if (set.count(*it)) { return true; }
    set.insert(target - *it);
  }

  return false;
}

bool Solution::findTarget(TreeNode* root, int k) {
  // we need a way to keep track of numbers in the tree, and also its a BST
  // so we can use the ordering to our benefit somehow

  // we can check if k == 0 (this means we found our thing)
  // else if k - root.val > root.val we can check right subtree, else left
  // subtree and if the root ends up being null we return false

  // we can use extra space, perform a bfs, and run the same algo as twosum
  // problem

  queue<TreeNode*> queue;
  queue.push(root);

  vector<int> nodes;

  while (queue.size() != 0) {
    int size = queue.size();
    for (int i = 0; i < size; i++) {
      TreeNode* curr = queue.front();
      queue.pop();

      if (curr != nullptr) {
        nodes.push_back(curr->val);
        if (curr->left != nullptr) { queue.push(curr->left); }
        if (curr->right != nullptr) { queue.push(curr->right); }
      }
    }
  }
  return Solution::twoSumBool(nodes, k);
}

bool Solution::isOneBitCharacter(vector<int>& bits) {
  // for this one, if we are at 10 or 11, that is 2 bits
  // and we need to skip those two
  // 0 represents one bit

  // so given a vector of bits we should loop through
  // if bits[i] = 1, then this means we are either at 10 or 11, so we skip the
  // next two bits if bits[i] = 0, it is a one bit, and we move forward 1

  // if i = length, and we incremented by 2, then that means the last 0 is not 1
  // bit so we return false else we can return true

  // so at worst, if i == bits.size() - 1, then that means we return true
  // else return false? since if the second last is 1, then we increment by 2
  // and i becomes >= size but if second last bit is 0, and we know last bit is
  // always 0, we increment by 1 both times, loop still breaks and i becomes
  // equal to size()

  int i = 0;
  int n = bits.size();
  while (i < n - 1) {
    if (bits[i] == 0) {
      i += 1;
    } else {
      i += 2;
    }
  }

  return i == n - 1;
}

bool Solution::judgeCircle(string moves) {
  int U = 0;
  int D = 0;
  int L = 0;
  int R = 0;
  for (auto it = moves.begin(); it != moves.end(); ++it) {
    switch (*it) {
      case 'U':
        U++;
        break;
      case 'D':
        D++;
        break;
      case 'L':
        L++;
        break;
      default:
        R++;
        break;
    }
  }

  return U == D && L == R;
}

vector<vector<int>> Solution::imageSmoother(vector<vector<int>>& img) {
  // for each element in the vector, we need to calculate the floor average of
  // its neighbors and set it to that value

  int m = img.size();
  int n = img[0].size();

  // this is initializing a vector of size mxn to return, where each value is 0
  vector<vector<int>> retVect(m, vector<int>(n, 0));

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) { retVect[i][j] = smoothen(img, i, j); }
  }

  return retVect;
}

int Solution::smoothen(vector<vector<int>>& mat, int x, int y) {
  // we need to calculate average of this and if another direction exists, it
  // too and return the average the other directions are -10, 00, 10 0-1, 01 and
  // 11 (if they exist) so lets loop through those i,j values, calculating
  // direction of x and y adding it to our running sum, and then getting
  // avergate

  // x and y represent curr

  int m = mat.size();
  int n = mat[0].size();

  int sum = 0;
  int count = 0;

  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int dx = x + i;
      int dy = y + j;

      // we check if its a valid direction
      if ((dx >= 0 && dx < m) && (dy >= 0 && dy < n)) {
        sum += mat[dx][dy];
        count++;
      }
    }
  }

  return floor(sum / count);
}

int Solution::findFinalValue(vector<int>& nums, int original) {
  unordered_set<int> set;

  for (vector<int>::iterator it = nums.begin(); it != nums.end(); it++) {
    set.insert(*it);
  }

  while (set.find(original) != set.end()) {
    // set.contains(original) is only there in cpp 20;
    // set.find returns an iterator to the value, else returns set.end()
    original *= 2;
  }

  return original;
}

int Solution::findSecondMinimumValue(TreeNode* root) {
  // we know based off how the tree is constructed, that root is the minimum
  // value we need to find the minimum next greatest value and if either subtree
  // doesn't have that, return null

  return dfsVal(root, root->val);
}

int Solution::dfsVal(TreeNode* root, int rootVal) {
  if (root == nullptr) { return -1; }

  if (root->val > rootVal) { return root->val; }

  int left = dfsVal(root->left, rootVal);
  int right = dfsVal(root->right, rootVal);

  if (left == -1) { return right; }

  if (right == -1) { return left; }

  return min(left, right);
}

int Solution::findLengthOfLCIS(vector<int>& nums) {
  // this is simple sliding window problem
  // strictly continuous means we have to go left to right, and break if there
  // is a fault
  if (nums.size() == 1) {
    return 1;  // this is because one element means that is the lcis
  }

  // given a dynamic window from i to j, if we find a violation, we must update
  // i = j and j to i + 1 but before that we have to update the max sequence
  // length with its prev value, and j - i + 1; why that? because say i = 0, and
  // j = 1, and we reach a violation the maximum length thus far should be 2 (0
  // and 1 inclusive)

  int i = 0;
  int maxSequence = 1;  // subsequence length must atleast be 1
  int j = 1;

  while (j < nums.size()) {
    if (nums[j - 1] < nums[j]) {
      j++;
    } else {
      // there is a fault
      maxSequence = max(maxSequence, j - i);
      i = j;
      j++;
    }
  }
  maxSequence = max(maxSequence,(int)nums.size() - i);  // this can be because last loop we realize entire array is properly increasing
  return maxSequence;
}

bool Solution::validPalindrome(string s){
  //same idea as regular, but we get to skip one and continue
  bool skipped = false;
  int i = 0;
  int j = s.size() - 1;
  while(j > i){ 
    if(skipped && s.at(i) != s.at(j)){ //this means we already 'deleted' one
      return false;
    }
    if(s.at(i) != s.at(j) && !skipped){ //this means we will choose to delete this one
      skipped = true;
    }
    //continue iteration of two pointer
    i++;
    j--;
  }
  return true;
}