package algorithms.searching;

public class LinearSearch {
    public void main(String[] args) {
        int[] arr = new int[] {1,2,4,5,18,20};
        System.out.println(linearSearch(arr, 18));
    }

    private boolean linearSearch(int[] arr, int target){
        for(int a: arr){
            if(a == target){
                return true;
            }
        }
        return false;
    }
}
