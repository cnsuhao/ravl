
#include "Ravl/Threads/ConditionalMutex.hh"
#include "Ravl/OS/Date.hh"
#include <iostream>

int testConditionalMutex();

int main()
{
  int ln = 0;
  if((ln = testConditionalMutex()) != 0) {
    std::cerr<< "Test failed at " << ln << "\n";
    return 1;
  }
  std::cout << "Passed. \n";
  return 0;
}

int testConditionalMutex() {

  RavlN::ConditionalMutexC condMutex;

  condMutex.Lock();

  const float theDelay = 3.0;

  {
    // Test WaitUntil.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(condMutex.WaitUntil(theDeadline)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }

  {
    // Test Wait() with fixed delay.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(condMutex.Wait(theDelay)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }

  condMutex.Unlock();
  return 0;
}
