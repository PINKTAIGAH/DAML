#ifndef MyClass_H
#define MyClass_H

class MyClass{
    public:
        // Declare constructor
        MyClass(int inputOne);

        // Declare a public method 
        int testMethod(int inputTwo);

        void changeStoredValue(int input);

    private:
        // Declare private variable
        int storedValue;
    
    // Dont forget semi colon on class declarations
};

#endif