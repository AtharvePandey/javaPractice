// // stack vs heap OOP concepts

// /*
//     Video questions below:

//     1.) Explain the key differences between stack and heap memory in cpp. Which
//    is faster and why?
//         - Stack memory is for statically declared variables, either on global
//    scope or local function scope; can be primitive data types or objects
//         - Heap memory is for pointers, or anything dynamically allocated for,
//    via new keyword or malloc()/alloc()
//         - Stack memory is faster for allocation and deallocation because its
//    just moving stackPtr as memory is allocated or deallocated
//         - Heap memory is slower beacuse you need to find a suitable free block
//    of memory (done by memory manager) 2.) In what situations would you need
//    dynamic object creation over static give examples
//         - we can use dynamic memory allocation when we want an object outside
//    the lifetime of some function, or if we don't know much of the object
//         - i.e a function which returns a smart pointer to a dynamically
//    allocated vector
//         - statically allocated memory can be used for local variables who have
//    no need outside of function scope

// */

// #include <iostream>

// #include "./L1.cpp"

// Employee
//     e1;  // this is statically defining an object on the call stack (globally)

// int temp() {
//   Employee e2;  // this is doing it locally for that function also on stack
//   Employee* e3 =
//       new Employee();  // this is heap memory, all pointers will be stored there

//   // we have to delete allocated memory manually, if static declared, gets
//   // deleted when out of scope (stack)

//   // static memory might cause overflow, and dynamic allocated memory might have
//   // memory leakage
//   main();
//   return 0;
// }
