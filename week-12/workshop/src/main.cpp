// Include the new class
#include "MyClass.h"

#include <iostream>
#include <vector>

void excersiceOne(){

    int classInput {17};
    int methodInput {12};
    int changeInput {20};
    // Make an intrance of a class
    MyClass classInstance(classInput);
    // Run class method
    int result = classInstance.testMethod(methodInput);
    std::cout << "The result was " << result << std::endl;

    classInstance.changeStoredValue(changeInput);

    result = classInstance.testMethod(methodInput);
    std::cout << "The result was " << result << std::endl;
}

void excersiceTwo(){
    int classInput {17};
    int methodInput {12};
    int changeInput {20};

    // Make a pointer to an instance of a class
    MyClass* anInstancePointer = new MyClass(classInput);
    int result = anInstancePointer->testMethod(methodInput);
    std::cout << "The result was " << result << std::endl;

    // Delete object referenced by pointer
    delete anInstancePointer;
    // Set dangling pointer to Null
    anInstancePointer = NULL;
}

void excersiceThree(){
    // Iterate over required range
    for (int i{0}; i<10; ++i){
        
        if (i == 5){
            std::cout << "hello" << std::endl;
            continue;
        }

        std::cout << i << std::endl;
    }
}

void excersiceFour(){
    // Initialise vector
    std::vector<int> myVector{};

    for (int i{0}; i<10; ++i){
        // Append number 
        myVector.push_back(i);
    }

    std::cout << myVector.size() << std::endl;
    long unsigned int length {myVector.size()};

    for (int i:myVector){
        // Print vector value
        std::cout << i;
        
        // Print comma if loop has not ended
        if (i != (length-1)){
            std::cout << ", ";
        }
    }

    // Make new line
    std::cout << std::endl;
}

void excersiceFive(){
    int total{0};

    for (int i{0}; i<2000; ++i){
        
        if (i%3 == 0){
            total += i;
        }
        else if (i%5 == 0){
            total += i;
        }
        else if (i%7 == 0){
            total += i;
        }

    }
    // Print out result
    std::cout << "The sum of multiples 3, 5 and 7 is " << total << std::endl;
}

void excersiceSix(){
    int total {0};
    int limit {2000000};
    // Define the first two values of the fibbonaci sequence
    int previous {0};
    int current {1};

    while (current < limit){
        if (current%2 != 0){
            total += current;
        }
        
        // Calculate new value of sequence
        int newValue = previous + current;

        // Update values for next iteration
        previous = current;
        current = newValue;
    }

    std::cout << "The sum of the first 20000000 odd values in the fibbonaci sequence is " << total << std::endl;
}

// Define main function

int main(/*int argc, char * argv[] */){
    
    // excersiceOne();
    // excersiceTwo();
    // excersiceThree();
    // excersiceFour();
    // excersiceFive();
    excersiceSix();

    return 0;
}