package algorithms.searching;

public class BinarySearch {
    public void main(String[] args) {
        int[] arr = new int[] {1,2,4,5,18,20}; //array has to be sorted in some form
        System.out.println(binarySearch(arr, 18));
    }

    private boolean binarySearch(int[] arr, int target){
        int low = 0;
        int high = arr.length-1;
        while(low<high){
            int mid = (low + high)/2;
            if(arr[mid] == target){
                return true;
            }else if(arr[mid] < target){//search the right half of the array
                low = mid + 1;
            }else{ //search left half of array
                high = mid;
            }
        }
        return false;
    }
}
