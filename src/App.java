import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;
import java.util.TreeSet;
//import java.util.function.Function;
import java.util.stream.Collectors;

//import java.util.stream.Stream;
//import java.util.stream.Stream;
import java.util.Random;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
//import java.util.Collection;
import java.util.Collections;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class App {
    private static App app = new App(); // to test the methods

    public static void main(String[] args) throws Exception {
        String s = "abcabcbb";
        app.lengthOfLongestSubstring(s);
    }

    public ListNode tempfunction1() {
        ListNode head = new ListNode(4);
        head.next = new ListNode(5);
        head.next.next = new ListNode(7);
        return head;
    }

    public ListNode tempfunction2() {
        ListNode head = new ListNode(2);
        head.next = new ListNode(3);
        head.next.next = new ListNode(5);
        return head;
    }

    // given an array of numbers, return a list of all duplicates in the array:
    // should be in O(1) space, and O(n) time, all numbers in array are from 1 --> n
    // where n is size of array

    public ArrayList<Integer> duplicates(int[] arr) {
        // there is a way to do this in O(n) time and space if we use a hashmap to get a
        // count of all numbers that appear more than once
        // and then return an arraylist of all keys whos value > 1
        // but since we are constrianed for O(1) space (not including the list to
        // return), and since all numbers are 1 --> n inclusive
        // we can cleverly use the array itself to see if a number repeats by matching
        // value at index with the actual index
        // if that new index is negative we add it to the arraylist else we can just
        // make it negative and move on

        ArrayList<Integer> arrayList = new ArrayList<>();
        for (int i = 0; i < arr.length; i++) { // iterate through the array
            int index = Math.abs(arr[i]) - 1; // current index is number in array - 1, since numbers are 1 --> n
                                              // inclusive
            if (arr[index] < 0) { // if that index value is 0, it means we already visited it, which means the
                                  // value is a duplicate
                arrayList.add(Math.abs(arr[index]) + 1); // so we add it to the arraylist
            } else {
                arr[index] = -arr[index]; // else we make it a negative number
            }
        }
        return arrayList;

    }

    public static boolean containsDuplicates(int[] nums) { // returns true if there are duplicates in array or false
                                                           // otherwise
        HashSet<Integer> hs = new HashSet<>();
        for (int i = 0; i < nums.length; i++) {
            if (hs.contains(nums[i])) {
                return true;
            } else {
                hs.add(nums[i]);
            }
        }
        return false;
    }

    public static int[] twoSum(int[] nums, int sum) { // given an array of nums, return an array of indices that sum to
                                                      // sum
        HashMap<Integer, Integer> hm = new HashMap<>();
        int[] retArr = new int[2];
        for (int i = 0; i < nums.length; i++) {
            // we want to return the indexes, so store index as value with key being the
            // difference
            if (hm.containsKey(nums[i])) {
                retArr[0] = hm.get(nums[i]);
                retArr[1] = i;
                return retArr;
            } else {
                hm.put(sum - nums[i], i);
            }
        }
        return null;
    }

    public static int longestContinuousSubsequence(int[] nums) {
        // we need to write an O(n) time solution that returns a number which is the
        // longest subsequence in an array continuous
        // order of elements in starting array doesnt matter
        // e.g nums = [1,4,5,2,3,100] --> ans 5 because 1,2,3,4,5 is formed before 100
        // breaks up the subsequence, and is the longest sequence also
        // duplicates dont count,
        if (nums.length == 0) {
            return 0;
        } // edge case where array is empty and we dont have a subsequence

        int longestSoFar = 0; // a variable to store longest subsequence
        // since duplicates dont count, we can just store every element in a hashset

        Set<Integer> hs = new HashSet<>();

        // fill in the hs
        for (int num : nums) {
            hs.add(num);
        }
        // now we need to iterate through the hashset and figure out how to find a
        // longest subsequence
        for (int num : hs) {
            // we can start counting a subsequence if the number is the smallest in that
            // subsequence,
            // i.e if we dont have a number smaller than that number

            if (!hs.contains(num - 1)) {
                // so if this number is the smallest in the subsequence
                int currNum = num;
                int currStreak = 1; // we know we have a streak of atleast size 1
                // next is to count the numbers in the rest of the sequence, and update the
                // longest sequence
                while (hs.contains(currNum + 1)) {
                    currNum += 1;
                    currStreak += 1;
                }
                longestSoFar = Math.max(currStreak, longestSoFar);
            }
        }
        return longestSoFar;

    }

    public static int romanToInt(String s) {
        // Dictionary of Roman numerals
        Map<Character, Integer> map = new HashMap<>();
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);

        // This variable will store result
        int res = 0;
        int prevValue = 0;

        // Loop for each character from right to left
        for (int i = s.length() - 1; i >= 0; i--) {
            int currentValue = map.get(s.charAt(i));
            // If the current value is less than the previous value, subtract it
            if (currentValue < prevValue) {
                res -= currentValue;
            } else {
                res += currentValue;
            }
            prevValue = currentValue;
        }

        return res;
    }

    public static void merge(int[] nums1, int m, int[] nums2, int n) {
        // 3 pointers, one is i = m - 1, other is j = n - 1, and k points to m + n - 1
        // so if arr1 is [1,2,3,0,0,0], k points to index 5 (0), i points to index 2
        // (3), and j
        // will point to [2,5,6] 6 in the second array (index 2)
        // since second array elements are what we need to move into the first array, we
        // need a while loop until j is
        // >=0
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        while (j >= 0) {
            if (i >= 0 && nums1[i] > nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }

        }
    }

    public static int removeElement(int[] nums, int val) {
        int i = 0;
        for (int j = 0; j < nums.length; j++) {
            if (nums[j] != val) {
                int temp = nums[i];
                nums[i] = nums[j];
                nums[j] = temp;
                i++;
            }

        }
        return i;
    }

    public static int removeDuplicates(int[] nums) { // [1,1,2,3,3]
        int i = 1;
        // start at index 1 for both i and j, since 0th index itself isnt a duplicate
        for (int j = 1; j < nums.length; j++) {
            if (nums[j] != nums[j - 1]) { // simply move a nonduplicate value into duplicate spot
                nums[i] = nums[j]; // so if duplicate dont do this
                i++; // increment the spot of the new index up to which no duplicates exist
            }
        }
        return i;
    }

    public static int remove2Duplicates(int[] nums) { // to keep 2 duplicates
        int i = 0; // start at start of array because in this one we count 2 duplicates
        for (int e : nums) {
            if (i == 0 || i == 1 || nums[i - 2] != e) {// if we are at the first 2 elements or the element 2 before is
                                                       // not current number, move current number to that index
                nums[i] = e;
                i++;
            }
        }
        return i; // this simply moves all non 2 duplicates to the end, to actually get the
                  // result, Array.splice(0,i);
    }

    public int majorityElement(int[] nums) {
        int candidate = nums[0]; // default max element
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] == candidate) {
                count++;
            } else {
                count--;
            }
            if (count == 0) {
                candidate = nums[i];
            }
        }
        return candidate;
    }

    public class KthLargest {
        // class to return kth largest element in an array after we add an element to it
        // size is guaranteed to be bound by k
        // note a minheap is when the root is always the smallest compared to children
        // we always use a priority queue to implement a min heap
        private PriorityQueue<Integer> pq = new PriorityQueue<>(); // using a min heap
        private int k;

        public KthLargest(int k, int[] nums) {
            this.k = k;
            for (int num : nums) {
                this.pq.add(num);
                if (this.pq.size() > this.k) {
                    this.pq.poll(); // to keep the size at k
                }
            }
        }

        public int add(int a) {
            // will add a to the 'array' and then see if size is > k, if it is remove, and
            // then return kth largest
            this.pq.add(a);
            // return this.pq.size() > this.k ? this.pq.poll() : this.pq.peek(); //this
            // logic won't work because we only want to return kth largest
            if (this.pq.size() > k) {
                this.pq.poll();
            }
            return this.pq.peek();
        }
    }

    public int findKthLargest(int[] nums, int k) {
        // the goal is to find the kth largest number in an unsorted array without
        // actually sorting the array
        // do it in linear time

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        for (int num : nums) {
            maxHeap.add(num);
        }
        for (int i = 1; i < k; i++) {
            maxHeap.poll();
        }
        return maxHeap.peek();
    }

    // public class RecentCounter {
    // private Queue<Integer> q;

    // public RecentCounter() {
    // q = new LinkedList<>();
    // }

    // public int ping(int t) {
    // q.add(t);
    // while (q.peek() < t - 3000) {
    // q.poll();
    // }
    // return q.size();
    // }
    // }
    public class Q {
        // implement queue behavior using 2 stacks
        private Stack<Integer> in;
        private Stack<Integer> out;

        public Q() {
            this.in = new Stack<>();
            this.out = new Stack<>();
        }

        public void push(int x) {
            this.in.push(x);
        }

        public int pop() {
            if (this.out.empty()) {
                while (!this.in.empty()) {
                    this.out.push(this.in.pop());
                }
                return this.out.pop();
            }
            return this.out.pop();
        }

        public int peek() {
            if (this.out.empty()) {
                while (!this.in.empty()) {
                    this.out.push(this.in.pop());
                }
                return this.out.peek();
            }

            return this.out.peek();
        }

        public boolean empty() {
            return this.in.empty() && this.out.empty();
        }
    }

    class Solution {
        // the trick for this is n - blacklist.length; [0,2,3] is blacklist,
        // blacklist.length is 3
        // n is 6, so we will randomly pick numbers 0 --> 5
        // the trick is having a temporary whitelist which is filled with numbers from n
        // - blacklist.length --> n - 1 inclusive
        // in this case we fill whitelist set from 6 - 3 --> 6 - 1; or 3 --> 5; {3,4,5}
        // is temp whitelist
        // then for every number in blacklist, if its in temp whitelist, remove it from
        // tempwhitelist
        // after this construct actual whitelist as a hashmap as such:
        // for every number in blacklist, if that number is < n - blacklist.length
        // (0,1,2), then map it to a number
        // that is in tempwhitelist, here use an iterator for the whitelist set...

        private int n; // --> consider n = 6
        private int[] blacklist; // --> consider [0,2,3]
        private HashMap<Integer, Integer> whiteList = new HashMap<>();
        private Random rand;

        public Solution(int n, int[] blacklist) {
            this.n = n;
            this.blacklist = blacklist;
            this.rand = new Random();
            // step1, make a set tempWhitelist that has numbers from n-blacklist.length to n
            // exclusive
            Set<Integer> tempWhiteList = new HashSet<>();
            for (int i = this.n - this.blacklist.length; i < this.n; i++) {
                tempWhiteList.add(i);
            }
            // next remove any element in tempwhitelist that is also in the blacklist array
            for (int i : blacklist) {
                if (tempWhiteList.contains(i)) {
                    tempWhiteList.remove(i);
                }
            }
            // after this we need to make a hashmap, where for every number in blacklist,
            // if that number is lessthan n - blacklist.length (so 0,1,2) in this case, then
            // add key whose value is first number in twl
            // iterate through the hashset and give each key,value pair
            Iterator<Integer> iter = tempWhiteList.iterator();
            for (int num : blacklist) {
                if (num < this.n - blacklist.length) {
                    this.whiteList.put(num, iter.next());
                }
            }
            // hashmap is {(0,4) (2,5)} , note 3 is not there because it isn't < n -
            // blacklist.length
        }

        public int pick() {
            // in here, generate a random number from 0 --> n - blacklist.length exclusive
            // so (0,1,2)
            // if number is in whitelist, then just return the value, else we can just
            // return the number...
            int random = rand.nextInt(n - blacklist.length);
            return this.whiteList.containsKey(random) ? this.whiteList.get(random) : random;
        }
    }

    /**
     * Your Solution object will be instantiated and called as such:
     * Solution obj = new Solution(n, blacklist);
     * int param_1 = obj.pick();
     */

    public int minAsciiDelete(String s1, String s2) {
        // Given two strings s1 and s2, return the lowest ASCII sum of deleted
        // characters to make two strings equal.
        // OPT(i,j) = ascii(i) + OPT(i--,0) if s2.length is 0
        // OPT(i,j) = ascii(j) + OPT(0,j--) if s1.length is 0

        // above two are basecases, since we need to consider cases where either string
        // is empty
        // this is similar to the string matching thing we did in cs311

        // OPT(i,j) = OPT(i--,j--) or min(ascii(i) + OPT(i--, j) , ascii(j) +
        // OPT(i,j--))
        // here we choose OPT(i--,j--) if both the characters are equal, else we take
        // the minimum, since we want to find the minimum
        // i.e choose the smallest possible ascii value to add and move forward
        // considering "all" possible cases at once.

        int[][] dp = new int[s1.length() + 1][s2.length() + 1];
        dp[0][0] = 0; // if two strings are empty.

        // we init the array with the base cases
        // if s2 is empty:
        for (int i = 1; i < s1.length(); i++) {
            dp[i][0] = dp[i - 1][0] + s1.charAt(i - 1); // dp[i-1][0] is substring which stores values of previous ascii
                                                        // character, and we add curr
        }
        // if s1 is empty
        for (int j = 0; j < s2.length(); j++) {
            dp[0][j] = dp[0][j - 1] + s2.charAt(j - 1); // same logic as above
        }

        // next we have to populate the dp array character by character based off 2
        // cases, if the characters are equal or not
        for (int i = 0; i < s1.length(); i++) {
            for (int j = 0; j < s2.length(); j++) {
                if (s1.charAt(i) == s2.charAt(j)) {
                    dp[i][j] = dp[i--][j--];
                } else {
                    dp[i][j] = Math.min(s1.charAt(i - 1) + dp[i--][j], s2.charAt(j - 1) + dp[i][j--]);
                }
            }
        }
        return dp[s1.length()][s2.length()];
    }

    // linked list stuff below;
    public class ListNode {
        int val;
        ListNode next;

        public ListNode() {

        }

        public ListNode(int val) {
            this.val = val;
        }

        public ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // merge 2 sorted lists together
    public ListNode merge(ListNode l1, ListNode l2) {
        if (l1 == null && l2 == null) {
            return null;
        }
        ListNode tempHead = new ListNode();
        ListNode lst = tempHead;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                lst.next = l1;
                l1 = l1.next;
            } else {
                lst.next = l2;
                l2 = l2.next;
            }
            lst = lst.next;
        }
        if (l1 != null) {
            lst.next = l1; // rest of l1 is rest of lst
        } else {
            lst.next = l2;
        }
        return tempHead.next; // this is the first actual node of the new list
    }

    public ListNode removeDuplicates(ListNode head) {
        // Establish two pointers
        ListNode slow = head;
        ListNode fast = head;

        // While two pointers are not null
        while (fast != null) {
            // Re-assign 'next' of each node to a node with a different value
            while (fast != null && slow.val == fast.val) {
                fast = fast.next;
            }
            slow.next = fast;
            // Move both pointers to the node with a different value
            slow = fast;
        }

        // Return the head node
        return head;
    }

    public boolean cycle(ListNode head) {
        // given head of a linked list, we need to see if there is a cycle
        // since list size is constrained by 1e4, if we iterate more than that, there is
        // a cycle
        // Instantiate two pointers where slow points to the head of the list and fast
        // points to head
        ListNode slow = head;
        ListNode fast = head;

        // Iterate over the LL (while slow != fast)
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            // If slow is ever equal to fast then the while loop condition is broken and
            // cycle is found.
            if (slow == fast)
                return true;
        }
        // If exit loop return False
        return false;
    }

    public ListNode whereCycle(ListNode head) {
        // instead of confirming if there is a cycle, we need to return the node where
        // the cycle starts
        // for this we get when slow = fast
        // after start at head again, and move both slow and head to next
        // until slow = dummy head
        // point of intersection is the start of the cycle
        ListNode slow = head;
        ListNode fast = head;
        ListNode tempHead = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                break;
            }
        }
        if (!(fast == null || fast.next == null)) { // as in we aren't at the end of the linked list
            while (tempHead != slow) {
                tempHead = tempHead.next;
                slow = slow.next;
            }
        } else {
            return null;
        }
        return tempHead;

    }

    public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
        // Get the lengths of both lists
        int lengthA = getLength(headA);
        int lengthB = getLength(headB);

        // Make the starting points equal in terms of distance to the end of the lists
        while (lengthA > lengthB) {
            headA = headA.next;
            lengthA--;
        }

        while (lengthB > lengthA) {
            headB = headB.next;
            lengthB--;
        }

        // Move both pointers until they intersect
        while (headA != headB) {
            headA = headA.next;
            headB = headB.next;
        }

        return headA; // This will be null if there is no intersection
    }

    private static int getLength(ListNode head) {
        return head == null ? 0 : 1 + getLength(head.next);
    }

    // given head of linked list, remove the nth node from the end
    // [1,2,3,4,5] and n = 2, remove 2nd element from end, 4 and return the linked
    // list
    public ListNode removeNth(ListNode head, int n) {
        ListNode temp = new ListNode(0);
        ListNode first = temp;
        ListNode second = temp;
        temp.next = head;

        for (int i = 0; i <= n + 1; i++) {
            first = first.next;
        }

        while (first != null) {
            second = second.next;
            first = first.next;
        }

        second.next = second.next.next;

        return temp.next;
    }

    public ListNode addList(ListNode h1, ListNode h2) {
        ListNode dummyListNode = new ListNode(0);
        ListNode currList = dummyListNode;
        // currlist is a node we will use to build our result sum
        // consider list 234 and 567
        // to add we would do 2 + 5, 6 + 3 and 7 + 4 with remainder, 1 added to end
        // 7 + 4 = 11 = sum; sum % 10 is the next digit, 1, and sum/10 is the remainder
        // calculated
        int remainder = 0;
        while (h1 != null || h2 != null || remainder != 0) { // go until both lists are null and there is no more
                                                             // remainder
            int firstNum = h1 == null ? 0 : h1.val;
            int secondNum = h2 == null ? 0 : h2.val;
            int sum = remainder + firstNum + secondNum;
            currList.next = new ListNode(sum % 10); // add one's place digit to next node
            remainder = sum / 10; // gets the tens digit for next iteration of loop

            h1 = h1 == null ? null : h1.next;
            h2 = h2 == null ? null : h2.next;
            currList = currList.next;
        }
        return dummyListNode.next;
    }

    public ListNode addListII(ListNode h1, ListNode h2) {
        // in this one the numbers aren't passed in reversed, and we can't reverse the
        // numbers either
        // we can probably come up with a recursive approach to this problem.
        // a simpler approach would be to just use a stack, it doesnt count as reversing
        // a list, and we can still get the answer

        // LITERALLY THE IDEA IS SAME AS ADDLIST, JUST USE A STACK TO "REVERSE" THE
        // GIVEN LISTS

        int remainder = 0;
        int sum = 0;
        ListNode result = null;
        // since we are using a stack for this problem writing a helper method is easier
        Stack<Integer> h1Num = app.stackify(h1);
        Stack<Integer> h2Num = app.stackify(h2);

        while (!h1Num.empty() || !h2Num.empty() || remainder != 0) {
            int num1 = h1Num.empty() ? 0 : h1Num.pop();
            int num2 = h2Num.empty() ? 0 : h2Num.pop();
            sum = remainder + num1 + num2;
            // calculate remainder for next iteration
            remainder = sum / 10;
            ListNode node = new ListNode(sum % 10);
            node.next = result;
            result = node;
        }
        return result;
    }

    private Stack<Integer> stackify(ListNode head) {
        Stack<Integer> retStack = new Stack<>();
        while (head != null) {
            retStack.push(head.val);
            head = head.next;
        }
        return retStack;
    }

    public class palendrome {
        public ListNode tempHead;

        public boolean isPalendromeList(ListNode head) {
            // return if a list is a palendrome or not; 1 - 2 - 2 - 1 is a palendrome list
            // have a node at the start of the list, and we recursively check from the end
            // of the list
            tempHead = head;
            return helper(tempHead);
        }

        private boolean helper(ListNode currNode) {
            if (currNode != null) {
                // here if the currnode is not null, as in we still have to go to the end of the
                // list then go to end of the list
                // since the function returns a boolean however, we have to put it in an if
                // statement
                if (!helper(currNode)) {
                    return false; // as in our recursive call somehow returns false in the call stack.
                }

                if (!(tempHead.val == currNode.val)) {
                    // if at any point the values aren't equal
                    return false;
                }
                tempHead = tempHead.next; // iterate through the next node once the call stack goes through last node
                                          // and secondlast and so on

            }
            return true;
        }
    }

    public ListNode remDup2(ListNode head) {
        // return a list without any duplicates at all, and the list must be sorted
        // 1 - 2 - 3 - 3 - 4 - 4 - 5 would return 1 - 2 - 5
        // have 2 pointers, one starts at head.next, the other at head.next.next because
        // the first node itself can't be a duplicate?
        // have a third pointer that starts before head, returning 3rd.next as the final
        // answer

        ListNode temp = new ListNode(0);
        ListNode curr = head;
        ListNode prev = temp;
        prev.next = curr;
        while (prev != null && curr != null) {
            ListNode next = curr.next;

            if (next != null && curr.val == next.val) {
                while (next != null && curr.val == next.val) {
                    next = next.next;
                }
                prev.next = next;
                curr = next;
            } else {
                prev = curr;
                curr = next;
            }
        }
        return temp.next;
    }

    public ListNode[] splitListToParts(ListNode head, int k) {
        // the only reason this question is a medium is due to the maths part
        // given k the number of parts to split the list into, we need to try and split
        // it
        // such that each part is split as equally as possible into k parts

        // the size of each part will then be N/k if there are N nodes in the list.
        // the first N%k parts will have a size of N + 1 by observation alone

        int sizeOfEachPart = App.getLength(head) / k; // number of nodes per part
        int numberOfPartsWithExtraSize = App.getLength(head) % k; // --> first n%k parts have an extra node
        int count = 0; // how many nodes are in one part so far
        int numberOfPartsMadeSoFar = 0; // keep track of how many parts we have assembled
        ListNode listIterator = head; // where we start
        ListNode tempHead = head; // the current head --> what will actually iterate?
        ListNode[] retArr = new ListNode[App.getLength(head)]; // worst case, k = 1;

        // we want to go through the list until we reach sizeOfEachPart; until count =
        // sizeOfEachPart
        // then if numberOfPartsMadeSoFar = numberOfPartsWithExtraSize, we will add an
        // extra node to the current sublist, and set the next
        // value after adding to null
        // else just set the next value of last node to null
        // and after each part is assembled, move listPointer to next node, and put old
        // head pointer into the return array
        // add an index to array
        int i = 0;
        while (numberOfPartsMadeSoFar < k) {
            while (count < sizeOfEachPart) { // making the parts themselves
                listIterator = listIterator.next;
                ++count;
            }
            // now we have ending and starting for first part
            if (numberOfPartsMadeSoFar < numberOfPartsWithExtraSize) {
                // we need to add an extra node;
                listIterator = listIterator.next;
            }
            // next we need to add null to end of list, but cant use listIterator, and we
            // also need to add new head to array
            ListNode headSegment = tempHead;
            ListNode tailSegment = listIterator;
            tailSegment.next = null;
            retArr[i] = headSegment;
            tempHead = listIterator;
            numberOfPartsMadeSoFar++;
        }
        return retArr;
    }

    // Given two integer arrays nums1 and nums2, return the maximum length of a
    // subarray that appears in both arrays.

    public int findLength(int[] nums1, int[] nums2) {
        // this is like minAscii delete, except instead of strings we have arrays, and
        // instead of min we have max
        // we have a variable to keep track of max length
        // base case is 0 or 1 depending on if both arrays are length 1 and the first
        // numbers are equal
        // the length of the array is bounded from 1 -> 10^3

        int[][] dp = new int[nums1.length + 1][nums2.length + 1];
        dp[0][0] = (nums1.length == 1 && nums2.length == 1 && nums1[0] == nums2[0]) || nums1[0] == nums2[0] ? 1 : 0;
        int maxLength = 0;
        for (int i = 1; i < nums1.length; i++) {
            for (int j = 1; j < nums2.length; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1; // update dp
                    maxLength = Math.max(maxLength, dp[i][j]); // update the max length
                }
            }
        }
        return maxLength;
    }

    // Given an array of strings words representing an English Dictionary, return
    // the longest word in words that can be built one character at a time by other
    // words in words.

    // If there is more than one possible answer, return the longest word with the
    // smallest lexicographical order. If there is no answer, return the empty
    // string.

    // Note that the word should be built from left to right with each additional
    // character being added to the end of a previous word.

    public String longestWord(String[] words) {
        // for this problem, since we have to return the most earliest word in
        // alphabetical order
        // instead of keeping track of all possible candidates, it is easier to just
        // sort the array
        Arrays.sort(words);
        // next we need to consider a couple of cases
        // since all words are sorted, each word will either be less length or the final
        // candidate
        String finalCandidate = "";
        // we need a way of checking if we have already come across the current words
        // prefixes before making the current word our candidate
        // since a HashSet has constant storage and lookup, we can just use that
        HashSet<String> hs = new HashSet<>();
        // for each word in the array, we can only add it to the hashset if either the
        // word is length 1(in which case it is a def a building block)
        // or if there is a prefix of the current word somewhere in the hashset.
        // e.g if the word was some, then som should be in the hashset...that is
        // currWord.substring(0, currWord.length - 1)
        // we also need to update the finalcandidate string, but we can only that if it
        // is longer than previous candidate
        for (String currWord : words) {
            if (currWord.length() == 1 || hs.contains(currWord.substring(0, currWord.length() - 1))) {
                hs.add(currWord);
                if (currWord.length() > finalCandidate.length()) {
                    finalCandidate = currWord;
                }
            }
        }
        return finalCandidate;

    }

    public List<String> removeComments(String[] source) {
        List<String> li = new ArrayList<>();
        boolean multiLineComment = false;
        StringBuilder newLine = new StringBuilder();

        for (String line : source) {
            // we need to account for new line
            if (!multiLineComment) {
                newLine = new StringBuilder();
            }
            // next we need to build our new line and keep track of which lines to skip and
            // be wary of inline too

            int i = 0;
            while (i < line.length()) {
                if (!multiLineComment && i + 1 < line.length() && line.charAt(i) == '/' && line.charAt(i + 1) == '/') {
                    // Single line comment found
                    break;
                } else if (multiLineComment && i + 1 < line.length() && line.charAt(i) == '/'
                        && line.charAt(i + 1) == '*') {
                    // Start of multi-line comment
                    multiLineComment = true;
                    i++;
                } else if (multiLineComment && i + 1 < line.length() && line.charAt(i) == '*'
                        && line.charAt(i + 1) == '/') {
                    // End of multi-line comment
                    multiLineComment = false;
                    i++;
                } else if (multiLineComment) {
                    newLine.append(line.charAt(i));
                }
                i++;
            }

            if (!multiLineComment && newLine.length() > 0) {
                li.add(newLine.toString());
            }
        }
        return li;

    }

    // given an array of numbers, where value at each index represents how much you
    // can jump, return true or false depending on:
    // can we reach the end of the array based off where we can jump?

    public static boolean canJump(int[] nums) {
        int goal = nums.length - 1; // we want to reach the end of the array
        for (int i = nums.length - 1; i >= 0; i--) {
            // start from the end of the array, and use a greedy approach;
            // if we can reach the tile that is before the current goal, we can reach
            // current goal
            // loop until goal is set to its final value
            // if goal is 0 that means we can jump to end

            if (i + nums[i] > i) {
                goal = i;
            }
        }
        return goal == 0;
    }

    // given a string such as "hello how doing"
    // reverse it to get "doing how hello"

    public static String revWords(String s) {
        // s = s.trim();
        // String[] strArr = s.split("\s");
        // // now we just reverse the array
        // StringBuilder sb = new StringBuilder();
        // for (int i = strArr.length - 1; i >= 0; i--) {
        // sb.append(strArr[i]);
        // sb.append(" ");
        // }
        // return sb.toString().trim();

        // above is by using in built methods with O(1) time and space
        // we will most likely need to do this without inbuild methods
        // the trick to do this without inbuild methods is to tokenize each element

        // to tokenize all we need to do is go through the string, and like the split
        // method
        // add to a return list, reverse and return
        ArrayList<String> retList = new ArrayList<>();
        String temp = "";
        for (char c : s.toCharArray()) {
            if (c != ' ') {
                temp += c;
            } else if (!temp.isEmpty()) {
                retList.add(temp);
                temp = "";
            }
        }
        if (!temp.isEmpty()) {
            retList.add(temp);
            // add the remaining string if any exists
        }
        // next we have to reverse the array
        int l = 0;
        int r = retList.size() - 1;
        while (l < r) {
            String tem = retList.get(l);
            retList.set(l, retList.get(r));
            retList.set(r, tem);
            l++;
            r--;
        }
        return String.join(" ", retList);
    }

    public int binarySearch(int[] nums, int target) {
        // Initialize left and right pointers
        int low = 0;
        int high = nums.length - 1;

        // While left pointer is less than right pointer we have not exhausted the num
        // list
        while (low <= high) {
            // Get the mid point of the two pointers
            int mid = (low + high) / 2;

            // Check if mid point is less than, greater than, or equal to target

            // the number is equal, so we return the mid index
            if (nums[mid] == target)
                return mid;

            // mid point is less than target, then we know everything to the left of mid
            // point can be eliminated from search
            else if (target > nums[mid])
                low = mid + 1;

            // mid point is greater than target, then we know everything to the right of mid
            // point can be eliminated from search
            else
                high = mid - 1;
        }

        // The left pointer is greater than the right pointers, we have exhausted the
        // num list, return -1
        return -1;
    }

    // We are given an array asteroids of integers representing asteroids in a row.
    // For each asteroid, the absolute value represents its size, and the sign
    // represents its direction
    // (positive meaning right, negative meaning left). Each asteroid moves at the
    // same speed.
    // Find out the state of the asteroids after all collisions. If two asteroids
    // meet, the smaller
    // one will explode. If both are the same size, both will explode. Two asteroids
    // moving in the same direction will never meet.

    public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> tracker = new Stack<>();
        for (int asteroid : asteroids) {
            // This while loop handles multiple collisions
            while (!tracker.isEmpty() && asteroid < 0 && tracker.peek() > 0) {
                int top = tracker.peek();
                if (Math.abs(asteroid) > top) {
                    tracker.pop(); // Remove the top because the negative asteroid is stronger
                } else if (Math.abs(asteroid) == top) {
                    tracker.pop(); // Both asteroids are destroyed
                    asteroid = 0; // Current asteroid also destroyed
                } else {
                    asteroid = 0; // Current asteroid destroyed
                }
            }
            if (asteroid != 0) {
                tracker.push(asteroid); // Add the asteroid if it has not been destroyed
            }
        }

        // Convert stack to array
        int[] retArr = new int[tracker.size()];
        for (int i = retArr.length - 1; i >= 0; i--) {
            retArr[i] = tracker.pop();
        }

        return retArr;
    }

    // using binary search to find the pivot index of the array
    // if an array has been rotated from 0 -> n times, where n is length of the
    // array
    // we need to find the smallest value/pivot index of the array

    public int minIndex(int[] nums) {
        // we can use a modified version of binary search for this problem
        int low = 0;
        int high = nums.length;
        while (low < high) { // binary search boiler plate code
            int mid = (low + high) / 2;
            if (nums[mid] < nums[high]) {
                // this means the pivot index is somewhere in the left part of the array
                // this is because everything after the pivot index is increasing normally
                // but everything bigger than the pivot index should be to the left of it
                // which means we need to search the left half of the array
                high = mid;
            } else {
                low = mid + 1; // because that means we are before the pivot index
                // the numbers aren't increasing they are decreasing
                // search right half of array
            }
        }
        return nums[high]; // this is the value at pivot index
        // return high --> the pivot index itself
    }

    public int monotoneSequence(int n) {
        // given an integer, all we have to do is return the next biggest monotone
        // sequence
        // a number is a monotone sequence if from left to right, left <= right
        // e.g 124, 1234, 499, are all monotone sequences
        // 3214 is not a monotone sequence

        // the algorithm to return the next largest monotone sequence is as such:
        // find the first cliff (where num[i-1] > num[i])
        // i is the ith index of the number and is the cliff (inclusive)
        // after cliff is found, decrement every number before it by 1
        // iterate back to start of number (index 0)
        // once the start number is decremented by 1, change all numbers after start
        // number to 9

        // since the number passed in we can't iterate, we will convert into a String

        char[] num = String.valueOf(n).toCharArray();
        int i = 1;

        // first we need to find the cliff
        // even if multiple cliffs exists, we only need to find the first one
        // to find a cliff we iterate until either we run into the condition for a clif
        // or until we exceed length
        while (i < num.length && num[i - 1] <= num[i]) {
            i++;
        }

        // next we need to decrement every number before the cliff by 1
        while (i > 0 && i < num.length && num[i - 1] > num[i]) {
            num[i - 1] -= 1;
            i--;
        }

        // after this we need to make everything else a 9 after the last decremented
        // digit
        for (int j = i + 1; j < num.length; j++) {
            num[j] = '9';
        }

        // since we are returning an int, just convert the char array to an int
        return Integer.parseInt(String.valueOf(num));
    }

    public int maxProduct(int[] nums) {
        // we need to keep track of the max product so far, the min product so far
        // (because of negatives) and our result
        int max = nums[0];
        int min = nums[0];
        int result = nums[0];
        for (int i = 1; i < nums.length; i++) {
            // since max is updated before min, and we need to use the old max in
            // calculating/updating our current min
            int oldMax = max;
            // for each iteration, all three above variables will be updated, since we are
            // keeping track of them
            // the max will be updated if either current number is greater than max, or if
            // currNumber * max is greater than max
            // or if current number times the smallest (negative) number is greater than max
            max = Math.max(Math.max(max * nums[i], min * nums[i]), nums[i]);
            // our current min is also updated similarly as our max is, but we use the
            // oldMax
            min = Math.min(Math.min(oldMax * nums[i], min * nums[i]), nums[i]);
            // the result will only be updated if a new max exists
            if (max > result) {
                result = max;
            }
        }
        return result; // this is the largest multiplication from one of the subarrays
        // the solution doesn't actually return which subarray gives maximum product
    }

    public int maxSum(int[] nums) {
        // just like above except this time we need to find the max sum
        // the same logic is known as Kadanes algorithm
        // it is applicatble in many ways just like binary search

        int min = nums[0];
        int max = nums[0];
        int result = nums[0];

        for (int i = 1; i < nums.length; i++) {
            int oldMax = max;
            max = Math.max(Math.max(nums[i] + max, nums[i] + min), nums[i]);
            min = Math.min(Math.min(nums[i] + oldMax, nums[i] + min), nums[i]);
            if (max > result) {
                result = max;
            }
        }
        return result;
        // same logic but for addition instead of substraction
    }

    public int findPeakElement(int[] nums) {
        // says o(logn) time so they obv want a binary search like approach (we are also
        // SEARCHING for a peak)
        // like a min index property, this one will check if left and right are strictly
        // less than middle...

        // the array itself isn't sorted however, so we need to think of a way to choose
        // which half to eliminate...
        // we are told no two elements that are next to each other are equal, so
        // nums[mid] will be either the greatest
        // (best case) --> just return mid, or either left, right or both will be
        // greater.
        // in which case we need to decide which one to choose:
        // the idea is to choose the greater side if the mid is not the greatest, this
        // is because we are told
        // the ends of array are -infinity
        // we are basically garunteed a solution

        int low = 0;
        int high = nums.length - 1;

        while (low < high) {
            int mid = (low + high) / 2;
            if (nums[mid] > nums[mid + 1]) {
                high = mid;
            } else {
                low = mid + 1;
            }
        }
        return low;

    }

    // Given an array of strings words and an integer k, return the k most frequent
    // strings.
    // Return the answer sorted by the frequency from highest to lowest. Sort the
    // words with the same
    // frequency by their lexicographical order.
    public List<String> topKFrequentElements(String[] words, int k) {
        // idea is to use a minheap and also a hashmap
        // the hashmap is going to store key,value pairs to keep count of all the words
        Map<String, Integer> hm = new HashMap<>();
        for (String word : words) {
            hm.put(word, hm.getOrDefault(word, 0) + 1);
        }
        // now that we have counts of strings, we need to use a maxHeap, poulating it
        // with the first k frequent words
        // since we are using a priority queue, it is easier to just put the "negative"
        // of the count in a minheap, because making a maxheap
        // is too much work and I don't want to search anything up
        // Since a priority queue in java can only take one type, we can simply use the
        // tuples given by our map

        PriorityQueue<Map.Entry<String, Integer>> maxHeap = new PriorityQueue<>((a, b) -> {
            if (a.getValue().equals(b.getValue())) {
                return a.getKey().compareTo(b.getKey());
            }
            return b.getValue() - a.getValue();
        });

        // a better way for maxheap

        // PriorityQueue<Map.Entry<String, Integer>> maxHeap = new
        // PriorityQueue<>(Collections.reverseOrder());

        // the above will define the rules for when adding a (key, value) pair into
        // priority queue
        // we need to populate the pq
        for (Map.Entry<String, Integer> entry : hm.entrySet()) {
            maxHeap.offer(entry);
        }

        // next we need to return the first k pairs
        List<String> retList = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            retList.add(maxHeap.poll().getKey());
        }
        return retList;
    }

    public boolean hasAlternatingBits(int n) {
        // given a number, return true or false depending on if its binary
        // representation has alternating bits of 0 and 1
        // this is a very easy problem if we know how to do bit manipulations
        // essentially we will look at the number bit by bit meaning the number will
        // eventually reach 0
        while (n != 0) {
            int firstBit = n & 1; // the right most bit is when u take current number, and & it with 1 (bit
                                  // manipulation rules)
            int secondBit = (n >> 1) & 1; // right shift the current number to getrid of right most bit, and & again to
                                          // get the new right most bit
            if ((firstBit ^ secondBit) == 0) { // do an exclusive or, exactly 1 of the bits has to be 1, and the other 0
                return false; // return false if thats not the case (means not alternating)
            }
            n = n >> 1; // right shift number to progress through it
        }
        return true; // if the conditional never entered, and we went through the entire number
                     // return true
    }

    public int climbingStairs(int n) {
        if (n == 0 || n == 1) {
            return 1;
        }
        // n is the number of steps it takes to reach the top of the staircase
        // for each step, you can either take 1 or 2 steps
        // return the number of ways you can climb the staircase
        int dp[] = new int[n];
        dp[0] = 0;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            // you can either take 1 or 2 steps at each staircase (i)
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    public int lastStoneWeight(int[] stones) {
        // we choose the heaviest 2 stones in the array
        // if both weights are equal, both stones get discarded
        // else the lighter stone is destroyed, and we put (heavy - light) back into the
        // array

        // we could convert the entire array into a max heap, take 2 stones at a time,
        // and then put result
        // back into the max heap.
        // break the while loop if second stone doesn't exist, since we are done after
        // that

        // total time complexity will be O(n) since we are converting arr into heap
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        for (int stone : stones) {
            maxHeap.add(stone);
        }

        while (maxHeap.size() > 1) {
            int firstStone = maxHeap.poll();
            int secondStone = maxHeap.poll();
            if (firstStone == secondStone) {
                continue; // both stones are discarded
            } else if (firstStone > secondStone) {
                // discard second stone, and push onto heap the difference
                maxHeap.offer(firstStone - secondStone);
            } else {
                maxHeap.offer(secondStone - firstStone);
            }
        }
        return maxHeap.peek() == null ? 0 : maxHeap.poll();

    }

    class MyHashSet {
        int[] arr;

        public MyHashSet() {
            arr = new int[(int) 1e6];
        }

        public void add(int val) {
            arr[val] = 1;
        }

        public void remove(int val) {
            if (arr[val] == 1) {
                arr[val] = 0;
            }
        }

        public boolean contains(int val) {
            return arr[val] == 1;
        }
    }

    // for most algorithms that require merging objects by some property,
    // the below union find class is usefull for keeping track of things such as
    // common accounts

    public class UnionFind {
        public int[] parent;

        public UnionFind(int n) {
            this.parent = new int[n];
            for (int i = 0; i < n; i++) {
                parent[i] = i; // init every index from 0 - n (where n is number of objects)
                // to the object itself; (each obj is its own parent)
            }
        }

        public void union(int x, int y) {
            // unionize 2 objects (give them same parent if they don't have one)
            int parentX = getParent(x);
            int parentY = getParent(y);

            if (parentX != parentY) {
                // pick one to be the parent
                this.parent[parentX] = parentY;
            }

        }

        public int getParent(int currObject) {
            if (this.parent[currObject] == currObject) {
                return currObject; // the parent is itself, hasnt been unionized yet...
            }
            // else we have to find the parent of the current object, and naturally
            // parent of any object is an object that is parent to itself;
            return getParent(this.parent[currObject]);
        }
    }

    // Given a list of accounts where each element accounts[i] is a list of strings,
    // where the first element accounts[i][0] is a name,
    // and the rest of the elements are emails representing emails of the account.
    // Now, we would like to merge these accounts. Two accounts definitely belong to
    // the same person if
    // there is some common email to both accounts. Note that even if two accounts
    // have the same name, they may
    // belong to different people as people could have the same name. A person can
    // have any number of accounts initially,
    // but all of their accounts definitely have the same name. After merging the
    // accounts, return the accounts in the
    // following format: the first element of each account is the name, and the rest
    // of the elements are emails in
    // sorted order. The accounts themselves can be returned in any order.

    public List<List<String>> accountsMerge(List<List<String>> accounts) {
        // return a new 2d list of type string which has all accounts belonging to the
        // same person as one account
        // i.e if 2 seperate elements in account list has atleast 1 shared email
        // account, then merge those two seperate accounts
        // into one account, obv getting rid of any duplicate email ID's

        // so first we need to keep track of which accounts are gonna have common
        // parents, and we can use a unionfind object
        // as defined above to do that

        UnionFind uf = new UnionFind(accounts.size()); // by union find algo def, we need to pass in a number, in this
                                                       // case, accounts.size

        // using a map, for each account in the 2d list, we have to put into map by two
        // conditions:
        // the map will be string, int type where int is the id (index of current
        // account in 2d array)
        // the string will be email
        // for each element from index 1 --> sublist.size, add with id.
        // if email already exists for some future account, merge those two accounts
        // using unionfind object

        HashMap<String, Integer> emailIdMap = new HashMap<>();
        for (int accountNumber = 0; accountNumber < accounts.size(); accountNumber++) {
            List<String> account = accounts.get(accountNumber);
            List<String> emailsForCurrentAccount = account.subList(1, account.size()); // since 0 index of list is name
                                                                                       // of account
            // next we have to add email,id pairs into map and follow 2 conditions;
            // if email doesn't exists in map, add it and account number, else unionize this
            // email's acc number with previous email's account number
            for (String email : emailsForCurrentAccount) {
                if (!emailIdMap.containsKey(email)) {
                    emailIdMap.put(email, accountNumber);
                } else {
                    int prevAccNum = emailIdMap.get(email);
                    uf.union(prevAccNum, accountNumber);
                }
            }
        }

        // now that we have a mapping of all accounts and their parents, we need to
        // return a new list
        // go through each account, find the direct parent and merge

        HashMap<Integer, Account> accountsWithRoot = new HashMap<>();
        for (int accId = 0; accId < accounts.size(); accId++) {
            List<String> account = accounts.get(accId); // for each account
            // get the direct parent, and if the direct parent is not in hashset, add it
            // with the current account
            // else get all emails and add those to the current account
            int currAccParent = uf.getParent(accId);

            if (!accountsWithRoot.containsKey(currAccParent)) {
                accountsWithRoot.put(currAccParent, new Account(accounts.get(currAccParent).get(0)));
            }
            List<String> emailList = account.subList(1, account.size());
            accountsWithRoot.get(currAccParent).emails.addAll(emailList);
        }

        return accountsWithRoot.values().stream().map(acc -> acc.toStringList()).collect(Collectors.toList());

    }

    public class Account {
        String name;
        Set<String> emails = new TreeSet<>((String a, String b) -> {
            return a.compareTo(b);
        });

        public Account(String name) {
            this.name = name;
        }

        public List<String> toStringList() {
            List<String> retList = new ArrayList<>();
            retList.add(name);
            retList.addAll(emails);
            return retList;
        }
    }

    // The school cafeteria offers circular and square sandwiches at lunch break,
    // referred to by numbers 0 and 1 respectively.
    // All students stand in a queue. Each student either prefers square or circular
    // sandwiches.
    // The number of sandwiches in the cafeteria is equal to the number of students.
    // The sandwiches are placed in a stack. At each step: If the student at
    // the front of the queue prefers the sandwich on the top of the stack, they
    // will take it and leave the queue. Otherwise, they will leave it and go
    // to the queue's end.This continues until none of the queue students want to
    // take the top sandwich and are thus unable to eat.
    // You are given two integer arrays students and sandwiches where sandwiches[i]
    // is the type of the i​​​​​​th sandwich in the stack
    // (i = 0 is the top of the stack) and students[j] is the preference of the
    // j​​​​​​th student in the initial queue (j = 0 is the front of the queue).
    // Return the number of students that are unable to eat.

    public int countStudents(int[] students, int[] sandwiches) {
        // its important to note that the order in which students recieve their
        // sandwhich doesn't matter
        // simply count the number of sandwhich types to student types
        // and then calculate who doesn't get any food
        int zero = 0;
        int one = 0;
        for (int student : students) {
            if (student == 0)
                zero++;
            else
                one++;
        }
        for (int sandwhich : sandwiches) {
            if (sandwhich == 0) {
                if (zero == 0) {
                    break;
                }
                zero--;
            } else {
                if (one == 0) {
                    break;
                }
                one--;
            }
        }
        return zero + one;
    }

    // There is a restaurant with a single chef. You are given an array customers,
    // where customers[i] = [arrivali, timei]:
    // arrivali is the arrival time of the ith customer. The arrival times are
    // sorted
    // in non-decreasing order. timei is the time needed to prepare the order
    // of the ith customer.When a customer arrives, he gives the chef his order,
    // and the chef starts preparing it once he is idle. The customer waits till the
    // chef finishes preparing his order. The chef does not prepare food for more
    // than one customer at a time. The chef prepares food for customers in the
    // order
    // they were given in the input.
    // Return the average waiting time of all customers.
    // Solutions within 10^-5 from the actual answer are considered accepted.

    public double averageWaitingTime(int[][] customers) {
        double total = 0; // divide this by length of the array to get answer
        int currentTime = 0; // this is to keep track of the current time
        for (int[] customer : customers) {
            // for each customer we can check 2 things
            // if the time customer arrives is greater than the current time
            // we need to wait until the time are equal before starting that cutomers order
            // else we start the customers order since they arrived while we were completing
            // an order
            if (currentTime < customer[0]) {
                // we wait
                currentTime = customer[0];
            }
            // else we have to update the total wait time for this customer, and also the
            // total time
            currentTime = currentTime + customer[1]; // --> the time it takes to cook this customers dish
            total += currentTime - customer[0]; // --> this is the amount of time this customer had to wait
        }
        return total / customers.length;
    }

    // You are given a string s of even length. Split this string into two halves of
    // equal lengths, and let a be the first half and b be the second half.

    // Two strings are alike if they have the same number of vowels ('a', 'e', 'i',
    // 'o', 'u', 'A', 'E', 'I', 'O', 'U'). Notice that s contains uppercase and
    // lowercase letters.

    // Return true if a and b are alike. Otherwise, return false.

    public boolean halvesAreAlike(String s) {
        // we can use regexs in java
        Pattern regex = Pattern.compile("[aeiou]", Pattern.CASE_INSENSITIVE);
        String a = s.substring(0, s.length() / 2);
        String b = s.substring(s.length() / 2, s.length());
        Matcher forA = regex.matcher(a);
        Matcher forB = regex.matcher(b);
        int left = 0;
        int right = 0;

        while (forA.find()) {
            left++;
        }
        while (forB.find()) {
            right++;
        }
        return left == right;
    }

    public boolean anagram(String s, String t) {
        HashMap<Character, Integer> countS = new HashMap<>();
        HashMap<Character, Integer> countT = new HashMap<>();
        if (s.length() != t.length()) {
            return false;
        }
        for (int i = 0; i < s.length(); i++) {
            countS.put(s.charAt(i), countS.getOrDefault(s.charAt(i), 0) + 1);
            countT.put(t.charAt(i), countT.getOrDefault(t.charAt(i), 0) + 1);
        }
        return countS.equals(countT);
    }

    // below is leetcodes solution

    // public boolean halvesAreAlike(String s) {
    // int count1 = 0, count2 = 0;

    // // Convert the string to lowercase
    // s = s.toLowerCase();

    // // Iterate through the first half of the string and count vowels
    // for (int i = 0; i < s.length() / 2; i++) {
    // if (s.charAt(i) == 'a' || s.charAt(i) == 'e' || s.charAt(i) == 'i' ||
    // s.charAt(i) == 'o' || s.charAt(i) == 'u') {
    // count1++;
    // }
    // }

    // // Iterate through the second half of the string and count vowels
    // for (int i = s.length() / 2; i < s.length(); i++) {
    // if (s.charAt(i) == 'a' || s.charAt(i) == 'e' || s.charAt(i) == 'i' ||
    // s.charAt(i) == 'o' || s.charAt(i) == 'u') {
    // count2++;
    // }
    // }

    // // Check if count1 is equal to count2
    // return count1 == count2;
    // }

    // You are given a string allowed consisting of distinct characters and an array
    // of strings words.
    // A string is consistent if all characters in the string appear in the string
    // allowed.
    // Return the number of consistent strings in the array words.

    public int countConsistentStrings(String allowed, String[] words) {
        int retCount = 0;
        Pattern regex = Pattern.compile(allowed, Pattern.CASE_INSENSITIVE);
        Matcher regexMatcher = null;
        for (String word : words) {
            regexMatcher = regex.matcher(word);
            if (regexMatcher.find()) {
                retCount++;
            }
        }
        return retCount;
    }

    // Below are all problems from Neetcode 150
    // Already done contains duplicate, anagrams, twoSum, start from the 4/150 in
    // the arrays and hashing section

    // given an array of strings, add similar anagrams to a sublist, and add that to
    // the return list
    // You should aim for a solution with O(m * n) time and O(m) space, where m is
    // the number of strings and n is the length of the longest string.
    // O(m) space means if there are 7 strings then we should have a space of 7
    // maps; return value isn't counted in this

    // map key,value where key is, instead of a hashmap which represents <char,
    // countChar>, a String that represents count of chars, and value what is
    // an empty list

    public List<List<String>> groupAnagrams(String[] strs) {
        HashMap<String, List<String>> hm = new HashMap<>();
        // so how would we make a string that represents the count of letters?
        // e.g if word is cab, then the stirng would be 111000...000 where length is 26
        // and 1 represents count
        // change string to a char array, and use [char - 'a']++ to map,
        for (String s : strs) {
            int[] count = new int[26]; // since there are 26 letters in the alphabet
            for (char c : s.toCharArray()) {
                count[c - 'a']++; // count at this index gets incremented by 1, where c - 'a' acts as a mapping
            }
            // once we have a count, we have to convert it into a key (string), add it to
            // the map if it doesnt exist, and then add the string to that list
            // so first convert it to a key
            String key = Arrays.toString(count);
            // add to map if it doesnt exist
            hm.putIfAbsent(key, new ArrayList<String>());
            // then add the string to that arrayList
            hm.get(key).add(s);
        }

        return new ArrayList<>(hm.values());
    }

    // for this one, given an array, just return an array of the k most frequent
    // elements
    // e.g if k is 2, then return an array of all the elements that appear atleast
    // twice

    // we can return the elements in any order
    // if order did matter, then i would use a minHeap, heapify the array and then
    // just go through the heap in a single pass
    // keeping track of curr elem and count (adding to the ret SET (no duplicates)
    // if it appears k or more times)
    // reset and repeating for each new element that shows up

    // because order doesn't matter, a hashmap would do

    // in a single pass, init count of each element, then for each key,value if
    // value >= k add to retSet
    // then return the set

    // You should aim for a solution with O(n) time and O(n) space, where n is the
    // size of the input array.
    public int[] topKFrequent(int[] nums, int k) {
        // Create a map to count the frequency of each number
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        // Create a min-heap (priority queue) to keep the top k frequent elements
        // The heap is ordered by the frequency of elements (smallest frequency at the
        // top)
        PriorityQueue<int[]> heap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
            heap.offer(new int[] { entry.getValue(), entry.getKey() });
            if (heap.size() > k) {
                heap.poll(); // Remove the element with the smallest frequency
            }
        }

        // Extract the elements from the heap
        // The result array will contain the k most frequent elements
        int[] res = new int[k];
        for (int i = 0; i < k; i++) {
            res[i] = heap.poll()[1];
        }
        return res;
    }

    // given a list of strings, write a program that encodes it, and then decodes
    // the strings and returns a list

    class EncodeDecode {
        // [a, word, in, this, array, #$%^]
        public String encode(List<String> strs) {
            // write a program here such that we can tokenize each string
            // and then use the same pattern to decode the string
            // using a map to encode is bad and memory consuming

            // any encoding algorithm works, but question wants no additional space in
            // general, so its better to store info about the string
            // within the string we need to figure out what to store
            // the length would make sense, and then how do we know when we have read one
            // word?
            // we need a way to figure out we have read a word, and also figure out how long
            // the word is

            // String retString = "";
            // Iterator<String> iter = strs.iterator();
            // while(iter.hasNext()){
            // String curr = iter.next();
            // curr += (curr.length() + '#');
            // retString += curr;
            // }

            // apparently the above code won't work because we aren't using a stringbuilder

            StringBuilder retString = new StringBuilder();
            for (String str : strs) {
                retString.append(str).append(str.length()).append("#");
            }

            return retString.toString();
        }

        public List<String> decode(String str) {
            // given a long string in format like : hello5#bye3#
            // we need to go through, on reading #, we need to take the previous n digits
            // (not including the number)
            // and then add that word to the retList
            // scan until the string has been gone through
            // a two pointer approach here would make sense

            // i starts at 0, j starts at i, j does the scanning till we reach the pound
            // at the pound symbol, j-1 is the number, and we take everything from i --> num
            // - 1, since length isnt 0 indexed
            // add that substring to list, and update i to be equal to j

            // i mean we dont even need the number?
            List<String> res = new ArrayList<>();
            int i = 0;
            while (i < str.length()) {
                int j = i;
                while (str.charAt(j) != '#') {
                    j++;
                }
                int length = Integer.parseInt(str.substring(i, j));
                i = j + 1;
                j = i + length;
                res.add(str.substring(i, j));
                i = j;
            }
            return res;
        }
    }

    // we need to write a function that returns a new array
    // at each index of the array is the product of the entire passed in array
    // except for the number that is at the index
    public int[] productExceptSelf(int[] nums) {
        // [1,2,3,4] returns new array [24, 12, 8, 6]
        // how do we go about doing this in O(n) time and space
        // the brute force way is to use 2 forloops and calculate product for i != j for
        // each index

        // another way is we calculate the product of the entire array, and then put
        // that number in each index
        // then loop through the array and for each index in nums, divide the num in the
        // new array
        // the above would have O(n) runtime and O(1) space

        int product = 1; // not zero since we will be multiplying
        // also if there are atleast 2 0's then we need to return an array of 0's since
        // the product will be 0
        // but if there is only 1 0, then we don't need to worry about anything since
        // that index will have a product
        int numberOfZeroes = 0;
        // so first lets calculate the total product of the array, unless a number is 0
        // this is so we can determine how many 0's there are and also the total product
        for (int num : nums) {
            if (num == 0) {
                numberOfZeroes++;
            } else {
                product *= num;
            }
        }
        if (numberOfZeroes > 1) { // this is because if there are 2 zeroes, then at any given product, we are
                                  // gonna have atleast 1 0 in its calculation
            return new int[nums.length];
        }

        int[] retArr = new int[nums.length]; // this doesn't factor in to the O(n) space complexity
        for (int i = 0; i < nums.length; i++) {
            // the product at index i is product/nums[i] if there are no zeros in the array
            if (numberOfZeroes > 0) {// if we have atleast 1 zero, then we have to be careful so that we don't divide
                                     // by 0
                if (nums[i] == 0) {
                    retArr[i] = product;
                } else {
                    retArr[i] = 0;
                }
            } else { // there are no 0's
                retArr[i] = product / nums[i];
            }
        }
        return retArr;

    }

    // the board is 9x9 and we have to validate:
    // 1 if there is any repeat in each row, --> map row number to a set, check for
    // duplicates
    // 2 column --> same logic as row
    // 3 and square
    // empty spots don't count, and numbers are 1-9
    // using a Map for each row, coloumn and then 3x3 subsquare to check would make
    // the most sense
    // the idea is to map each 3x3 subsquare to a coordinate e.g (1,1) represents
    // the top left sub-square
    // Input: board =
    // [["1","2","." ,".","3",".",".",".","."],
    // ["4",".","." ,"5",".",".",".",".","."],
    // [".","9","8" ,".",".",".",".",".","3"],

    // ["5",".",".",".","6",".",".",".","4"],
    // [".",".",".","8",".","3",".",".","5"],
    // ["7",".",".",".","2",".",".",".","6"],
    // [".",".",".",".",".",".","2",".","."],
    // [".",".",".","4","1","9",".",".","8"],
    // [".",".",".",".","8",".",".","7","9"]]

    // for the board above, i give example of what (1,1) will look like in the top
    // left
    // the indeces are 0,1,2 for the rows and columns, so to map those, and since
    // length of square is 3x3
    // we can say (row/3,col/3) is the current 3x3 square matrix coordinate
    // from there we retrieve its appropriate set (stored in the map) and check if
    // its in set or not

    public boolean isValidSudoku(char[][] board) {
        // so first initialize the Maps for row and column, we map respective number to
        // the set for that row/column
        Map<Integer, Set<Character>> row = new HashMap<>();
        Map<Integer, Set<Character>> column = new HashMap<>();
        // the map for when we check each subgrid will map the 3x3 coordinate and its
        // set
        Map<String, Set<Character>> gridMap = new HashMap<>();
        // next for each row, column in the board we can check
        for (int r = 0; r < board.length; r++) {
            for (int c = 0; c < board.length; c++) {
                String coordinate = r / 3 + "," + c / 3; // per the mapping described above
                if (board[r][c] == '.') {
                    continue; // since we don't care about empty spaces
                }
                if (row.computeIfAbsent(r, k -> new HashSet<>()).contains(board[r][c]) // the compute if absent
                                                                                       // basically init set if doesnt
                                                                                       // exist
                        || column.computeIfAbsent(c, k -> new HashSet<>()).contains(board[r][c]) || // else it will
                                                                                                    // check if that
                                                                                                    // number is in the
                                                                                                    // r/c/g
                        gridMap.computeIfAbsent(coordinate, k -> new HashSet<>()).contains(board[r][c])) {
                    return false; // if either of the things return true, then we will return false since number
                                  // alr exists
                }
                // so if none of the above hashsets have the number at r,c we add it to the set
                row.get(r).add(board[r][c]);
                column.get(c).add(board[r][c]);
                gridMap.get(coordinate).add(board[r][c]);
                // no need to add a hashset if one doesn't exist since the if clause does it for
                // us
            }
        }
        return true;
    }

    // Given an integer array of size n, find all elements that appear more than ⌊
    // n/3 ⌋ times.
    // Moores algorithm also works for this O(1) space, this one below is O(n) space
    // and time
    public List<Integer> majorityElementII(int[] nums) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for (int num : nums) {
            hm.put(num, hm.getOrDefault(num, 0) + 1);
        }

        List<Integer> result = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : hm.entrySet()) {
            if (entry.getValue() > nums.length / 3) {
                result.add(entry.getKey());
            }
        }
        return result;

    }

    // given an array of numbers, calculate and return a new array where the current
    // index holds the sum of this element and all prev ones
    // e.g [2,4,5,2,1] = [2,6,11,13,14] after function call

    public int[] prefixSum(int[] nums) {
        for (int i = 1; i < nums.length; i++) {
            nums[i] = nums[i - 1] + nums[i];
        }
        return nums;
    }
    // the above is a popular algorithm called prefix sum, and we can use it to
    // solve the following question

    // Given an array of integers nums and an integer k, return the total number of
    // subarrays whose sum equals to k.
    // A subarray is a contiguous non-empty sequence of elements within an array.

    public int subArraySum(int[] nums, int k) {
        // the trick here is to keep a count of prefix sums in a map
        // if we store prefix sum, and the count that sum appears, then we can keep
        // track of how
        // many subarrays exist that sum up to k
        // further more, we need to check if prefixsum(up to this index) - k is in the
        // map; the count of that will tell us # of subarrays
        // else just store prefix sum of count in the map
        int res = 0;
        HashMap<Integer, Integer> hm = new HashMap<>();
        int prefixSum = 0;
        hm.put(0, 1); // this is the start, since the first prefix sum would be 0
        for (int num : nums) {
            prefixSum += num;
            int diff = prefixSum - k;

            res += hm.getOrDefault(diff, 0);
            hm.put(prefixSum, 1 + hm.getOrDefault(prefixSum, 0));
        }
        return res;

    }

    // end of arrays and hashing section following is 2 pointer

    public void reverseString(char[] s) {
        // since its O(1) extra memory, no extra datastructures
        int i = 0;
        int j = s.length - 1;
        while (i < j) {
            swap(i, j, s);
            i++;
            j--;
        }
    }

    private void swap(int i, int j, char[] s) {
        char temp = s[i];
        s[i] = s[j];
        s[j] = temp;
    }

    public boolean isPalindrome(String s) {
        int l = 0, r = s.length() - 1;

        while (l < r) {
            while (l < r && !alphaNum(s.charAt(l))) {
                l++;
            }
            while (r > l && !alphaNum(s.charAt(r))) {
                r--;
            }
            if (Character.toLowerCase(s.charAt(l)) != Character.toLowerCase(s.charAt(r))) {
                return false;
            }
            l++;
            r--;
        }
        return true;
    }

    // is a string still palendromic if we remove a letter?
    public boolean isPalindromeII(String s) {
        int i = 0, j = s.length() - 1;

        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return isPalindrome(s, i + 1, j) || isPalindrome(s, i, j - 1);
            }
            i++;
            j--;
        }

        return true;
    }

    /* Check is s[i...j] is palindrome. */
    private boolean isPalindrome(String s, int i, int j) {

        while (i < j) {
            if (s.charAt(i) != s.charAt(j)) {
                return false;
            }
            i++;
            j--;
        }

        return true;
    }

    private boolean alphaNum(char c) {
        return (c >= 'A' && c <= 'Z' ||
                c >= 'a' && c <= 'z' ||
                c >= '0' && c <= '9');
    }

    // Given an array of integers numbers that is sorted in non-decreasing order.
    // Return the indices (1-indexed) of two numbers, [index1, index2], such that
    // they add
    // up to a given target number target and index1 < index2. Note that index1 and
    // index2 cannot be equal,
    // therefore you may not use the same element twice.
    // There will always be exactly one valid solution.
    // Your solution must use
    // O(1) additional space.
    public int[] twoIntegerSumII(int[] numbers, int target) {
        // o(1) space just means no DS
        // the array is sorted in increasing order so it makes sense
        // to use a two pointer approach where i is at 0, and j is at len-1
        // then we need to see when to increment i and j, if sum > target, then we
        // decrement j
        // else increment i
        // if i <= j then we have to break since we can't use the same index twice
        int l = 0;
        int r = numbers.length - 1;
        while (l != r) {
            int sum = numbers[l] + numbers[r];
            if (sum == target) {
                return new int[] { l + 1, r + 1 }; // since its a 1 indexed array assumption
            }
            if (sum > target) {
                r--;
            } else {
                l++;
            }
        }
        return null;
    }

    // given an array of nums, we need to return all triplets that sum up to 0
    // there should be no duplicate triplets
    // because there are no duplicates, we can just use a set to store triplets
    // we can also take inspiration from the code above

    // O(n^2) time should be the goal with O(1) space, so we don't need to use a
    // stack, just use retlist.contains

    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> retList = new ArrayList<>(); // this and the lists we push into it won't count as space
                                                         // since it is the return value
        // for this we can use the above approach, but since we are to return triplets,
        // we need to iterate through i, and use two pointer approach for j and k
        // since in the prev question nums was sorted, we need to do the same here
        Arrays.sort(nums); // no need to reassign to a variable since java is pass by value, and also since
                           // return type is void
        int length = nums.length;
        // next we run a for loop
        for (int i = 0; i < nums.length; i++) {
            // here we init j and k
            // per the above method, i was 0, and j was end of array
            // in this case since i is already there, j will have to be i + 1 and k will
            // still be end of array
            int j = i + 1;
            int k = length - 1;
            while (j < k) {
                if (nums[i] + nums[j] + nums[k] == 0) {
                    List<Integer> listToPush = new ArrayList<>();
                    listToPush.add(nums[i]);
                    listToPush.add(nums[j]);
                    listToPush.add(nums[k]);
                    if (!retList.contains(listToPush)) {
                        retList.add(listToPush);
                    }
                    j++;
                    k--;
                } else if (nums[i] + nums[j] + nums[k] < 0) {
                    j++;
                } else {
                    k--;
                }
            }
        }
        return retList;
    }

    // same idea as above, but with 4 integers instead of 2
    // all 4 integers need to be distinct, but the 4 different array pointers we use
    // already account for that
    public List<List<Integer>> fourSum(int[] nums, int target) {
        int length = nums.length;
        if (length < 4) {
            return null; // length has to be atleast 4 for this
        }
        List<List<Integer>> retList = new ArrayList<>();
        Arrays.sort(nums);
        for (int i = 0; i < length; i++) {
            for (int j = i + 1; j < length; j++) {
                int k = j + 1;
                int l = length - 1;
                while (k < l) {
                    long sum = (long) nums[i] + nums[j] + nums[k] + nums[l];
                    if (sum == target) {
                        List<Integer> listToPush = new ArrayList<>();
                        listToPush.add(nums[i]);
                        listToPush.add(nums[j]);
                        listToPush.add(nums[k]);
                        listToPush.add(nums[l]);
                        if (!retList.contains(listToPush)) {
                            retList.add(listToPush);
                        }
                        k++;
                        l--;
                    } else if (sum < target) {
                        k++;
                    } else {
                        l--;
                    }
                }
            }
        }
        return retList;

    }

    // You are given an integer array heights where heights[i] represents the height
    // of the ith bar
    // You may choose any two bars to form a container. Return the maximum amount of
    // water a container can store.

    public int maxArea(int[] heights) {
        // area = difference between i and j * height of the container
        // we want to get the max area which represents our container
        // with the most water
        // we are also returning the max area

        // e.g for first example, array is [1,7,2,5,4,7,3,6]
        // how do we know when to move the pointers
        // sorting won't work since its containers

        // start i at 0 and j at the end
        // then we can just move whichever pointer depending on the lower height
        // each iteration should keep track of the max area so far

        int maxAreaSoFar = 0;
        int i = 0;
        int j = heights.length - 1;
        while (i < j) {
            maxAreaSoFar = Math.max(maxAreaSoFar, (Math.abs(i - j) * Math.min(heights[i], heights[j]))); // update the
                                                                                                         // area
            if (heights[i] < heights[j]) {
                i++;
            } else {
                j--;
            }
        }
        return maxAreaSoFar;

    }

    // lc hard, do later
    public int trap(int[] height) {
        return 0;
    }

    // End of twopointer section, next is stack

    // given a string of paren, return if all parenthesis are valid or not
    // they must be opened and closed correctly

    public boolean isValid(String s) {
        // we use a stack, and also a Map to see which parenthesis maps to which
        Stack<Character> stack = new Stack<>();
        Map<Character, Character> closeToOpen = new HashMap<>();
        closeToOpen.put(')', '(');
        closeToOpen.put('[', ']');
        closeToOpen.put('{', '}');
        // next we need to check for each character, if its in the map, this means we
        // are at a close paren, and we need to pop an open one
        // so if an open one that maps correctly isnt there return false else true
        for (Character c : s.toCharArray()) {
            if (closeToOpen.containsKey(c)) {
                if (!stack.isEmpty() && closeToOpen.get(c).equals(stack.peek())) {
                    stack.pop();
                } else {
                    return false;
                }
            } else {
                stack.add(c);
            }
        }
        return stack.isEmpty();
    }

    class MinStack {
        // we need to design a stack class that also keeps track of the minimum number
        // in the stack
        // we have O(1) time and O(n) space, so we can use a stack to implement the
        // class

        Stack<Integer> stack;
        Stack<Integer> minStack;

        public MinStack() {
            this.stack = new Stack<>();
            this.minStack = new Stack<>();
        }

        public void push(int i) {
            // since we are pushing onto the stack, we need to see if this number is also
            // the min number
            this.stack.push(i);
            if (this.minStack.peek() > i) {
                this.minStack.push(i);
            } else {
                this.minStack.push(this.minStack.peek());
            }
        }

        public void pop() {
            if (!this.stack.empty())
                this.stack.pop();
            if (!this.minStack.empty())
                this.minStack.pop();
        }

        public int top() {
            if (!this.stack.empty())
                return this.stack.peek();
            else {
                return 0;
            }
        }

        public int getMin() {
            if (!this.minStack.empty())
                return this.minStack.peek();
            else {
                return 0;
            }
        }
    }

    public int evalRPN(String[] tokens) { // given array ["1","2","+","3","*","4","-"], evaluate output and return it.
                                          // here it is 5
        // O(n) time and space constraint
        Stack<Integer> stack = new Stack<>();
        for (String s : tokens) {
            if (s.equals("+")) {
                stack.push(stack.pop() + stack.pop());
            } else if (s.equals("-")) {
                int a = stack.pop();
                int b = stack.pop();
                stack.push(b - a);
            } else if (s.equals("*")) {
                stack.push(stack.pop() * stack.pop());
            } else if (s.equals("/")) {
                int c = stack.pop();
                int d = stack.pop();
                if (c != 0) {
                    stack.push(d / c);
                }
            } else {
                stack.push(Integer.parseInt(s));
            }
        }
        return stack.pop();
    }

    // given an array of strings, we need to find the longest common prefix among
    // all strings
    // similar to finding longestCommonPrefix in 2d array
    // array has the following in sorted form:
    // flight
    // flow
    // flower

    // take first and last word and then proceed to find the longest common prefix,
    // fl in this case between all 3 strings

    public String longestCommonPrefix(String[] strs) {
        Arrays.sort(strs);
        StringBuilder sb = new StringBuilder();
        int i = 0;
        while (strs[0].charAt(i) == strs[strs.length - 1].charAt(i)) {
            sb.append(strs[0].charAt(i));
        }
        return sb.toString();
        // will have a runtime of O(nlogn) since we are sorting the array, and will have
        // a space of O(n) because we are using stringBuilder object
    }

    // sort the array without using inbuilt methods
    // O(nlogn) time and O(n) space
    // its better if we use either merge sort or quicksort (merge is easier and
    // better so memorize that)

    public int[] sortArray(int[] nums) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
        for (int a : nums) {
            minHeap.add(a);
        }
        int i = 0;
        while (!minHeap.isEmpty()) {
            nums[i] = minHeap.poll();
            i++;
        }
        return nums;
    }

    // given array of 0,1,2 we need to sort them so that all 0's come first, then
    // 1's then 2's.
    // the goal is to sort in place and have O(1) space and O(n) runtime

    // we can take a three pointer approach here where i is at 0 and j is at len-1
    // and mid is at i
    // only because there are 3 numbers we can assume everything left of i and right
    // of j are already sorted, all we need to do is compare using mid, if value of
    // mid is 0, swap with i and increment i, else swap with j and decrement j
    // [2,0,2,1,1,0]
    public void sortColors(int[] nums) {
        int low = 0, mid = 0, high = nums.length - 1;
        while (mid <= high) {
            if (nums[mid] == 0) {
                swap(nums, low, mid);
                low++;
                mid++;
            } else if (nums[mid] == 1) {
                mid++;
            } else {
                swap(nums, mid, high);
                high--;
            }
        }
    }

    private void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    // You are given an integer array prices where prices[i] is the price of a given
    // stock on the ith day.
    // On each day, you may decide to buy and/or sell the stock. You can only hold
    // at most one share of the stock at any time.
    // However, you can buy it then immediately sell it on the same day.
    // Find and return the maximum profit you can achieve.

    // instead of a dp approach, a greedy approach is simpler
    // for each index, if the number at next index is bigger, add it to total profit
    // by saying prices[i+1] - prices[i] = maxProfit
    // its greedy because you are day-trading and trying to keep as much profit
    // possible, i.e selling everytime you make a bit of profit

    public int maxProfitII(int[] prices) {
        int maxProfit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i] < prices[i + 1]) {
                maxProfit += prices[i + 1] - prices[i];
            }
        }
        return maxProfit;
    }

    // given an array and an integer k, rotate each element k times to the right of
    // the array
    // [-1,-100,3,99] for each index, index + k, e.g if k is 2, then -1 goes to
    // index 2, -100 to index 3
    // 3 goes to index 4, but that is out of bounds; if we mod by length however,
    // then it goes to index 0 (4%4)
    // 99 goes to index 5 but we mod by length, 5%4 = 1;
    public void rotate(int[] nums, int k) {

        // so if im not wrong then the pattern is, (currIndex += k) % nums.length =
        // newIndex
        // here we already take care of currIndex if its less than length, since
        // anyThing % lessThanThatThing is thatThing

        // without following the O(1) extra space constraint, we can just use a new
        // array
        int length = nums.length;
        int[] arr = new int[length];
        for (int i = 0; i < length; i++) {
            int newIndex = (i + k) % length;
            arr[newIndex] = nums[i];
        }
        for (int i = 0; i < length; i++) {
            nums[i] = arr[i];
        }
    }

    // the solution which uses O(1) time but does it in place is very simple
    // reverse the entire array, then reverse the first k elements (subarray from (0
    // -> k]
    // then reverse the remaining elements (k to end]

    // start of the sliding window portion

    // Given an integer array nums and an integer k, return true if there are two
    // distinct indices
    // i and j in the array such that nums[i] == nums[j] and abs(i - j) <= k.

    // basically we want to find two numbers in an array that are equal, and the
    // distance between them is less than k

    public boolean containsNearbyDuplicate(int[] nums, int k) {
        // I still don't see how this is a sliding window problem
        // both solutions on leetcode use either a set or a map in O(n) time and space

        // the set approach feels more like sliding window but i like the hashmap one
        // better
        HashMap<Integer, Integer> hm = new HashMap<>();
        // where we store value at i, and also i
        for (int i = 0; i < nums.length; i++) {
            if (!hm.containsKey(nums[i])) {
                hm.put(nums[i], i);
            } else {
                // we check if the indexes are <= k apart
                if (Math.abs(hm.get(nums[i]) - i) <= k) {
                    return true;
                } else {
                    // we need to update the most recent index to i
                    hm.put(nums[i], i);
                }
            }
        }
        return false;
    }

    // You are given an array prices where prices[i] is the price of a given stock
    // on the ith day.
    // You want to maximize your profit by choosing a single day to buy one stock
    // and choosing a different
    // day in the future to sell that stock.
    // Return the maximum profit you can achieve from this transaction. If you
    // cannot achieve any profit, return 0.

    public int maxProfit(int[] prices) {
        // the obvious brute force solution is n^2 with constant space
        // where we run an embedded forloop
        // profit at day[i] = price[i] - priceWhenBought;

        // how is this one a sliding window i also don't know

        // obviously since we are figuring out when to buy and sell, we start off by
        // buying on day 1
        int buyDate = prices[0];
        int profitSoFar = 0; // since we haven't sold anything
        // next running a forloop, we can calculate the max profit, and also when to buy
        // now since its the stockmarket, we will buy on the cheapest day and sell on
        // the most expensive one
        for (int i = 1; i < prices.length; i++) {
            if (prices[i] < buyDate) { // if a cheaper day exists, then buy on that day
                buyDate = prices[i];
            } else {
                if (prices[i] - buyDate > profitSoFar) {
                    profitSoFar = prices[i] - buyDate;
                }
            }
        }
        return profitSoFar;
    }

    // Given a string s, find the length of the longest substring
    // without repeating characters.

    // now this one feels more like a 2 pointer problem
    public int lengthOfLongestSubstring(String s) {
        // the easy solution would be to use a set, and also keep count of 'how far we
        // explored into the string'
        // reset the count if duplicate letter comes up and also keep track of longest
        // count
        // single pass O(n) time and space
        int longestLengthSoFar = 0; // To store the result
        int i = 0; // Left pointer of the window
        Set<Character> seen = new HashSet<>(); // To track unique characters in the window

        for (int j = 0; j < s.length(); j++) { // j is the right pointer
            // If the character is already in the set, shrink the window from the left
            while (seen.contains(s.charAt(j))) {
                seen.remove(s.charAt(i));
                i++; // we basically shrink our window until we remove duplicate value
                // so that the new window won't have a duplicate value
            }

            // Add the current character to the set and update the longest length
            seen.add(s.charAt(j));
            longestLengthSoFar = Math.max(longestLengthSoFar, j - i + 1);
            // here the j-i+1 represents the current distance between 2 pointers, and we do
            // +1 since array is 0-indexed
        }

        return longestLengthSoFar;

        // i can see how this is sort of a sliding window approach since we have
        // difference between i and j be the window
        // and we dynamically update i to be j if conditions fail

    }

    // You are given a string s consisting of only uppercase english characters and
    // an integer k. You can choose
    // up to k characters of the string and replace them with any other uppercase
    // English character.
    // After performing at most k replacements, return the length of the longest
    // substring which contains only one distinct character.

    // this one is also sliding window, but we have a dynamic one instead of a fixed
    // one
    // O(n) space and time

    public int characterReplacement(String s, int k) {
        int result = 0;
        // the idea is we dont HAVE to replace anything, there are 26 characters in the
        // alphabet
        // we have a sliding window approach, and our goal is to replace at most k
        // characters in that window
        // such that the result is the largest it can be for that string

        // so what does our replacement look like? for a string ABABBA, and k = 2, we
        // can choose to replace either 2 A's or B's
        // such that the length of the substring is the longest and continuous: BBBBBA
        // makes longest substring length 5

        // but since we aren't actually replacing anything, how do we know when to
        // increase/decrease our window?
        // in any given instance of substring in S, it would make more sense to replace
        // the letter that occurs the least k times (if possible)
        // so for example, if from S = ABABBA, substring ABA, and k = 2, we can replace
        // at most 2 letters, here it would make most sense to
        // replace the B with A having longest subStr length of 3

        // we would need to keep a count of characters seen in a hashmap, and then start
        // our sliding window approach, where i and j start at 0
        // if in the windowlength (j-1+1) - mostFrequentCharCount <= k (since because we
        // would like to replace the least frequent character),
        // and since we can only replace k times, we need to make sure that in the
        // window the least frequent character occurs atmost k times
        // else we have to decrease our window, to do this we decrement the count of
        // character at left pointer and increase left pointer

        HashMap<Character, Integer> charCount = new HashMap<>();
        int i = 0;
        for (int j = 0; j < s.length(); j++) {
            charCount.put(s.charAt(j), (charCount.getOrDefault(s.charAt(j), 0)) + 1);
            if (((j - i) + 1) - Collections.max(charCount.values()) > k) {
                // we need to shrink window length, and decrement count of character in map at i
                charCount.put(s.charAt(i), charCount.get(s.charAt(i)) - 1);
                i++;
            }

            result = Math.max(result, (j - i) + 1); // (j-i) + 1 is the current window length
        }
        return result;
    }

    // start of tree section

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();

        inorder(root, res);
        return res;
    }

    private void inorder(TreeNode node, List<Integer> res) {
        if (node == null) {
            return;
        }
        inorder(node.left, res);
        res.add(node.val);
        inorder(node.right, res);
    }

    public List<Integer> preOrderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();

        preOrder(root, res);
        return res;
    }

    private void preOrder(TreeNode node, List<Integer> res) {
        if (node == null) {
            return;
        }
        res.add(node.val);
        preOrder(node.left, res);
        preOrder(node.right, res);
    }

    public List<Integer> postOrderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();

        postOrder(root, res);
        return res;
    }

    private void postOrder(TreeNode node, List<Integer> res) {
        if (node == null) {
            return;
        }
        postOrder(node.left, res);
        postOrder(node.right, res);
        res.add(node.val);
    }

    public int maxDepthOfTree(TreeNode root) {
        // the maximum depth is 0 if the root is null
        // if the root is non null, we return the maximum of the left and right subtrees
        return root == null ? 0 : 1 + Math.max(maxDepthOfTree(root.left), maxDepthOfTree(root.right));
    }

    public boolean treesAreEqual(TreeNode p, TreeNode q) {
        // here the key is to consider just one node first and then the recursive case
        // in terms of one node, for both trees, they are equal if both nodes are null
        // or if both nodes have the same value
        // then the recursive case is just to call on both nodes left and righ subtrees
        if (p == null && q == null) {
            return true;
        } else if (p == null || q == null) {
            return false;
        }
        return (p.val == q.val && treesAreEqual(p.left, q.left) && treesAreEqual(p.right, q.right));

    }

    public TreeNode invertBinaryTree(TreeNode root) {
        // for this one it is important to handle 1 case, and let recursion do the rest
        if (root != null) { // non leaf node (say we start at root of tree)
            TreeNode temp = root.right;
            root.right = root.left;
            root.left = temp;
            // the above is solving for one case
            // below is what the recursion handles;
            invertBinaryTree(root.left);
            invertBinaryTree(root.right);
        }
        return root;
    }

    // the diameter of a binary tree is, given any two nodes, the longest path in
    // the entire tree
    // obv the root to leaf is the longest path? wrong

    // the longest path doesn't have to go through the root
    // number of edges defines path length, so height of the current node is
    // basically number of edges to leaf node
    // we will use this idea to calculate the diameter
    /*
     * 1
     * \
     * 3
     * / \
     * 4 5
     * / \ / \
     * 6 7 8 9
     */

    // in the tree above, height of root is 3 since there are 3 edges to leaf from
    // root. i.e its the Max(left and right subtrees),
    // since the diameter is defined as longest path however, the diameter would be
    // 4 in this case. since the path from
    // 6-4-3-5-9 has 4 edges.

    // so null nodes have a height of 0, and diameter of current node is left +
    // right...which makes sense because trivially if we have
    // just one node, with 2 children, then the path would be left - root - right
    // where there are 2 edges and the diameter would be 2
    // height of node is different from the diameter, the height is max(left, right)
    // and then + 1 for height of parent node
    // and diameter is just height of left + right...

    public class DiameterOfTree {
        public int result = 0;

        public int diameter(TreeNode root) {
            height(root);
            return result;
        }

        private int height(TreeNode root) {
            if (root == null) {
                return 0; // this is where we will get a null leaf
            }
            int left = height(root.left);
            int right = height(root.right);
            result = Math.max(result, left + right); // to record the max diameter so far
            return 1 + Math.max(left, right); // return the height of the current node to parent node (to use to
                                              // calculate diameter.)

        }

    }

    // basically the key difference for the above, diameter is the max left + right
    // subtree height and the current diameter
    // take the height of the left subtree + right subtree gives diameter.

    // Given a binary tree, return true if it is height-balanced and false
    // otherwise.
    // A height-balanced binary tree is defined as a binary tree in which the left
    // and right
    // subtrees of every node differ in height by no more than 1.

    public boolean isBalanced(TreeNode root) {
        // basically get height of each individual subtree on left and right
        // return true if Math.abs(left-right) <= 1
        if (root == null) {
            return true; // An empty tree is balanced
        }

        // Calculate left and right subtree heights
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);

        // Check if the current node is balanced
        if (Math.abs(leftHeight - rightHeight) > 1) {
            return false;
        }

        // Recursively check if the left and right subtrees are balanced
        return isBalanced(root.left) && isBalanced(root.right);
    }

    public int height(TreeNode root) {
        if (root == null) { // trivial case where we reach leaf
            return 0;
        }
        int left = height(root.left);
        int right = height(root.right);
        return 1 + Math.max(left, right);
    }

    // given a tree, return a list that contains lists where each sublist contains
    // all nodes at that level
    public List<List<Integer>> levelOrderTraversal(TreeNode root) {
        List<List<Integer>> retList = new ArrayList<>();
        // BFS uses a queue to traverse the tree (when we search for an element) and we
        // go level by level
        // contrary to DFS which goes depth first using a stack
        Queue<TreeNode> queue = new ArrayDeque<>();
        queue.add(root); // add start node to queue
        while (!queue.isEmpty()) {
            // generally how bfs works check curr value, and if its not target, add children
            // to the queue
            // but we aren't searching for anything, we just need the values at each node in
            // its own subArr
            // im using a list since arrays are immutable and a pain in java
            List<Integer> subList = new ArrayList<>();
            for (int i = queue.size(); i > 0; i--) {
                TreeNode curr = queue.poll();
                if (curr != null) {
                    subList.add(curr.val);
                    queue.add(curr.left);
                    queue.add(curr.right);
                }
            }
            retList.add(subList);
        }
        return retList;

        // has O(n) time and space since we are visiting each node exactly once, and the
        // queue will have atmost n/2 elements at any given time

    }

    // Given a BST perform a zigzag traversal, i.e from root, visit first level
    // right to left, then second level left to right etc
    // return order of traversal in a list

    public List<List<Integer>> zigZagtraversal(TreeNode root) {
        // using the same logic as above, we have to traverse the tree in zigzag format
        // here, it depends on how we add the current level to the queue (left -> right
        // or right -> left)
        // since we add odd numbered levels in reverse, we can just keep track of level
        // currently...
        int levelNumber = 0;
        Queue<TreeNode> queue = new ArrayDeque<>();
        List<List<Integer>> retList = new ArrayList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> subList = new ArrayList<>();
            for (int i = queue.size(); i > 0; i--) {
                if (levelNumber % 2 == 0) {// we add to queue normally
                    TreeNode curr = queue.poll();
                    if (curr != null) {
                        subList.add(curr.val);
                        queue.add(curr.left);
                        queue.add(curr.right);
                    }
                } else {
                    TreeNode curr = queue.poll();
                    if (curr != null) {
                        subList.add(curr.val);
                        queue.add(curr.right);
                        queue.add(curr.left);
                    }
                }
            }
            retList.add(subList);
        }
        return retList;
    }

    // You are given the root of a binary tree. Return only the values of the
    // nodes that are visible from the right side of the tree, ordered from top to
    // bottom.

    public List<Integer> rightSideView(TreeNode root) {
        // I thought this was supposed to be simple, just recursively add everything you
        // see in root.right
        // but the problem can also have cases where there could exists a node on the
        // left subtree
        // that doesn't have a node blocking it on the right subtree
        // so you would add that node

        // the idea to solve this problem would be to do a level order traversal, and
        // add right most node in each level
        List<Integer> res = new ArrayList<>();
        Queue<TreeNode> q = new LinkedList<>();
        q.offer(root);

        while (!q.isEmpty()) {
            TreeNode rightSide = null;
            int qLen = q.size();

            for (int i = 0; i < qLen; i++) {
                TreeNode node = q.poll();
                if (node != null) {
                    rightSide = node;
                    q.offer(node.left);
                    q.offer(node.right);
                }
            }
            if (rightSide != null) {
                res.add(rightSide.val);
            }
        }
        return res;

    }

    // Given a binary search tree (BST) where all node values are unique, and two
    // nodes from the tree p and q, return
    // the lowest common ancestor (LCA) of the two nodes.
    // The lowest common ancestor between two nodes p and q is the lowest node in a
    // tree T such that both p and q
    // as descendants. The ancestor is allowed to be a descendant of itself.

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        // its like searching in a binary tree,
        // if we know p is less than root, and q is greater than root, then we just
        // return root
        // and we can sort of recursivley check for each subtree root, moving pointer
        TreeNode curr = root;
        while (curr != null) {
            if (p.val > curr.val && q.val > curr.val) {
                curr = curr.right;
            } else if (p.val < curr.val && q.val < curr.val) {
                curr = curr.left;
            } else {
                return curr;
            }
        }
        return null;
    }

    //given 2 arrays preorder and inorder traversals of a binary tree, we need to construct the tree
    //preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    //here preOrder[0] is topmost root, everything to left of that in inorder array is left subtree, and after is right...
    //so what would we pass to buildTree to implement the recursive soluton? 
    //well first we add this node, preorder[0], next we find that index in the inorder array to find out our left and right subtree size..
    //if index of preorder[0] in the inorder array is mid, then we can say everything to left of mid is the left subtree and right of mid is right
    //so for first iteration, mid = 1; leftsubtree would be inorder[0->mid] where mid is exclusive, and rightsubtree would be mid+1->length
    //where mid+1 is inclusive [start, end)... and for preorder, we start at 1, and go to mid (since 0 is this root, we take 1) and preorder
    //for right side is mid+1 to end

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        //basically we know preorder[0] will contain the first root (and every root of subtree)
        //everything to the left of inorder[indexOf(preorder[0])] is the left subtree, and everything else is the right subtree
        //so we can use a recursive approach where if preorder.length>0 set preorder[0] to root
        //then recursively call to build left and right subtrees

        TreeNode root = new TreeNode(preorder[0]);
        int mid = indexOf(inorder, preorder[0]);
        root.left = buildTree(Arrays.copyOfRange(preorder, 1, mid+1), Arrays.copyOfRange(inorder, 0, mid));
        root.right = buildTree(Arrays.copyOfRange(preorder, mid+1, preorder.length), Arrays.copyOfRange(inorder, mid+1, inorder.length));
        return root;

        
    }
    private int indexOf(int[] inorder, int target){
        for(int i = 0; i<inorder.length;i++){
            if(inorder[i] == target){
                return i;
            }
        }
        return -1;
    }

    // You are given the root node of a binary search tree (BST) and a value to
    // insert into the tree.
    // Return the root node of the BST after the insertion. It is guaranteed that
    // the new value does not exist in the original BST.
    // Notice that there may exist multiple valid ways for the insertion, as long as
    // the tree remains a BST after insertion.
    // You can return any of them.

    public TreeNode insertIntoBST(TreeNode root, int val) {
        if (root == null) {
            TreeNode node = new TreeNode(val);
            root = node;
            return root;
        }
        TreeNode curr = root;
        while (true) {
            if (curr.val < val) { // we serarch right half of the tree to add val to
                if (curr.right == null) {
                    TreeNode insert = new TreeNode(val);
                    curr.right = insert;
                    return root;
                } else {
                    curr = curr.right;
                }
            } else { // we search the left half of the tree
                if (curr.left == null) {
                    TreeNode insert = new TreeNode(val);
                    curr.left = insert;
                    return root;
                } else {
                    curr = curr.left;
                }
            }
        }
    }

    // Given a root node reference of a BST and a key, delete the node with the
    // given key in the BST.
    // Return the root node reference (possibly updated) of the BST.
    // Basically, the deletion can be divided into two stages:

    // Search for a node to remove.
    // If the node is found, delete the node.

    public TreeNode deleteNode(TreeNode root, int key) {
        // very complicated i would just memorize this one
        // makes no sense
        return null;
    }

    
    // Within a binary tree, a node x is considered good if the path from the root of the tree 
    // to the node x contains no nodes with a value greater than the value of node x
    // Given the root of a binary tree root, return the number of good nodes within the tree.
    
    class GoodNodes{
        int count = 0;
        public int goodNodes(TreeNode root){
            return dfs(root.val, root);
        }
    
        private int dfs(int maxSoFar, TreeNode root){
            if(root == null){ //null nodes are not good
                return 0;
            }
            if(root.val >= maxSoFar){
                this.count += 1; //update count since this node is good
            }
            maxSoFar = Math.max(maxSoFar, root.val); //update max
            //recursively traverse the tree with that paths max value...
            dfs(maxSoFar, root.left);
            dfs(maxSoFar, root.right);
            return this.count; //return the final global count
        }
    }

}
