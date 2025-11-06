// a class is a template or blueprint to represent an object

/*
    video questions below:

    1.) difference between an object and a class
        - a class represents an object, containing data memebers and function
   memebers (public or private)
        - an object however, is an instantiation of class, meaning it is a
   physical segmented memory where we can
        - assign attributes for that specifc object
    2.) data members vs. member functions
        - both can be public or private, members are primitive data types or
   objects belonging to a class
        - functions are things which get/set members, or they can do other
   things relating to the object 3.) implement an employee class, write a
   getter/setter for updating salary

*/

#include <stdio.h>

#include <iostream>

using namespace std;

class Employee {
 public:
  int salary;
  void updateSalary(int salary) { this->salary = salary; }
  int getSalary() { return this->salary; }

 private:
};

int main() {
  Employee e;
  e.salary = 5;  // we can only do this directly because salary is public
  e.updateSalary(12);
  cout << e.getSalary() << endl;
  return 0;
}
