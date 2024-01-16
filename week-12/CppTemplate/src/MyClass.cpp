// You have to include the class header here too
#include "MyClass.h"

// What the constructor does
MyClass::MyClass( int InputOne )
{
  m_storedValue = InputOne;
}

// What TestMethod does
int MyClass::TestMethod( int InputTwo )
{
  return InputTwo + m_storedValue;
}
