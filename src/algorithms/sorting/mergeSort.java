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
        ms.sort(null);
        System.out.println("sorted array: ");
        printArr.accept(arr);
    }

    private void sort(int[] arr){ //merge sort implementation
        //happens in 2 parts, we divide array into 2 parts
        //recursively merge the two parts
    }
}
