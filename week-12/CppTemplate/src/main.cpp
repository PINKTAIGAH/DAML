// Using the new class
#include "MyClass.h"

#include <iostream>

int main( /*int argc, char * argv[]*/ )
{
  // Create an instance of the class – just as we might have a variable that is an int, here we have a MyClass
  MyClass anInstance( 17 );

  // Run the method of MyClass and print the result
  int result = anInstance.TestMethod( 13 );
  std::cout << "The result was " << result << std::endl;
}
