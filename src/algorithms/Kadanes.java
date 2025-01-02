package algorithms;

public class Kadanes {
    public static void main(String[] args) {
        int[] nums = {-2,2,3,5,-4,2,3,-6,-7,3};
        kadanes(nums);
    }   
    
    public static int kadanes(int[] nums){
        //this algorithm is used to find the maximum subarray sum,
        //where given an array what is the max possible sum any of the elements can form? 
        //[-2,2,3,-1] --> summing all elements gives 2, but the max possible sum is 5

        int currentMaxSum = nums[0];
        int globalMaxSum = nums[0];
        for(int num : nums){
            currentMaxSum = Math.max(num, currentMaxSum + num);
            if(currentMaxSum > globalMaxSum){
                globalMaxSum = currentMaxSum;
            }
        }
        return globalMaxSum;
    }
}
