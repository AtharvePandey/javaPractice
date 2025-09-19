package algorithms.sorting;

import java.util.function.Consumer;

public class mergeSort {
    public static Consumer<int[]> printArr = arr -> {
        for (int i : arr) {
            System.out.print(i);
        }
    };
    private static mergeSort ms = new mergeSort();

    public static void main(String[] args) {
        System.out.println("unsorted array: ");
        int[] arr = { 2, 5, 3, 4, 1 };
        printArr.accept(arr);
        ms.mergeSort(null);
        System.out.println("sorted array: ");
        printArr.accept(arr);
    }

    private void mergeSort(int[] arr, int l, int r) { // merge sort implementation
        // happens in 2 parts, we divide array into 2 parts
        // recursively merge the two parts

        // how to divide the array in 2 parts recursively?
        // well divide this array in half first by calculating midpoint
        // then divide left and right subhalves
        // and then we can merge

        if (l < r) {
            int mid = l + (r - l) / 2;
            // now just call with left and right subhalves
            mergeSort(arr, 0, mid - 1);
            mergeSort(arr, mid + 1, r);

            //and then merge the halves
            merge(arr, l, mid, r);

        }
    }

    private void merge(int[] arr, int l, int mid, int r){
        //to merge lets make 2 arrays which represent each side
        //the size of these arrays will be the halves
        int arr1Size = mid-l + 1; //the left half, from left to mid + 1
        int arr2Size = r - mid; //the right half from mid to r technically
        
        //we can assume we have a full sized array although that is not the case
        //but it helps visualize the sizes of each subhalf

        int[] leftHalf = new int[arr1Size];
        int[] rightHalf = new int[arr2Size];

        //then lets copy fro main array into each half array
        for(int i = 0; i<arr1Size; i++){
            leftHalf[i] = arr[l + 1]; //why l+i?
        }

        for(int i = 0; i<arr2Size; i++){
            rightHalf[i] = arr[mid+1+i]; //why this? 
        }

        int i = 0, j = 0;
        int k = l; //this will be the index of our main arr while i,j will first and second

        while(i < arr1Size && j < arr2Size){
            if(leftHalf[i] < leftHalf[j]){
                arr[k] = leftHalf[i];
                i++;
            }else{
                arr[k] = rightHalf[j];
                j++;
            }
            k++;
        }

        while(i < arr1Size){
            arr[k] = leftHalf[i];
            i++;
            k++;
        }

        while(j < arr2Size){
            arr[k] = rightHalf[j];
            j++;
            k++;
        }
    }
}
