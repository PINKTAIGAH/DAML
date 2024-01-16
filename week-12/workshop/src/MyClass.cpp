#include "MyClass.h"

// Write class constructor
MyClass::MyClass(int inputOne){
    storedValue = inputOne;
}

// Write what test method does
int MyClass::testMethod(int inputTwo){
    return inputTwo + storedValue;
}

void MyClass::changeStoredValue(int input){
    storedValue = input;
}