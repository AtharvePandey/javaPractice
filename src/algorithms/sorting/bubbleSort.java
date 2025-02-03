package algorithms.sorting;

import java.util.function.Consumer;

public class bubbleSort {
    // basically for each element, swap 2 elements if n-1 element < n element
    private static bubbleSort bs = new bubbleSort();
    public static Consumer<int[]> consumer = x -> {
        for (int i = 0; i < x.length; i++) {
            System.out.print(x[i] + ", ");
        }
    };

    public static void main(String[] args) {
        System.out.println("unsorted array: ");
        int[] arr = { 2, 5, 3, 4, 1 };
        consumer.accept(arr);
        bs.bubblSort(arr);
        System.out.println("sorted array : ");
        consumer.accept(arr);

    }

    private void bubblSort(int[] arr) {
        // algorithm basically looks at each element, and swaps with next until can't
        // swap or this element is in order
        for (int i = 0; i < arr.length-1; i++) {
            for (int j = 0; j < arr.length-i-1; j++) { //we don't have 
                if (arr[j] > arr[j+1]) {
                    swap(j, j+1, arr);
                }
            }
        }

    }

    private void swap(int i, int j, int[] arr) { // swaps two indecies of an array
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}
