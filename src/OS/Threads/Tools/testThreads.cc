
#include "Ravl/Threads/ConditionalMutex.hh"
#include "Ravl/Threads/ThreadState.hh"
#include "Ravl/OS/Date.hh"
#include <iostream>

int testConditionalMutex();
int testThreadState();

int main()
{
  int ln = 0;
  if((ln = testConditionalMutex()) != 0) {
    std::cerr<< "Test failed at " << ln << "\n";
    return 1;
  }
  if((ln = testThreadState()) != 0) {
    std::cerr<< "Test failed at " << ln << "\n";
    return 1;
  }
  std::cout << "Passed. \n";
  return 0;
}

int testConditionalMutex() {

  RavlN::ConditionalMutexC condMutex;
  RavlN::MutexC mutex;
  mutex.Lock();

  const float theDelay = 1.0;

  {
    // Test WaitUntil.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(condMutex.WaitUntil(mutex,theDeadline)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }

  {
    // Test Wait() with fixed delay.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(condMutex.Wait(mutex,theDelay)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }

  mutex.Unlock();
  std::cout << "ConditionalMutex test passed. \n";
  return 0;
}

int testThreadState() {
  RavlN::ThreadStateC<int> state(0);

  const float theDelay = 1.0;
  {
    // Check wait
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(state.Wait(theDelay,1)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }
  {
    // Check static pass
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + 0.5;
    if(!state.Wait(theDelay,0)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt > theDeadline) return __LINE__;
  }
  {
    // Check wait
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    int newState = 1;
    if(state.Wait(theDelay,1,2,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }
  {
    // Check static pass conditions
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + 0.5;
    int newState = 1;
    if(!state.Wait(theDelay,1,0,newState)) return __LINE__;
    if(!state.Wait(theDelay,0,2,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt > theDeadline) return __LINE__;
  }
  {
    // Check wait.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    int newState = 1;
    if(state.Wait(theDelay,1,2,3,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }
  {
    // Check pass.
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + 0.5;
    int newState = 1;
    if(!state.Wait(theDelay,0,2,3,newState)) return __LINE__;
    if(!state.Wait(theDelay,1,0,3,newState)) return __LINE__;
    if(!state.Wait(theDelay,1,2,0,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt > theDeadline) return __LINE__;
  }
  {
    // Check wait
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    int newState = 1;
    if(state.Wait(theDelay,1,2,3,4,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;
  }
  {
    // Check pass
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + 0.5;
    int newState = 1;
    if(!state.Wait(theDelay,0,2,3,4,newState)) return __LINE__;
    if(!state.Wait(theDelay,1,0,3,4,newState)) return __LINE__;
    if(!state.Wait(theDelay,1,2,0,4,newState)) return __LINE__;
    if(!state.Wait(theDelay,1,2,3,0,newState)) return __LINE__;
    if(newState != 0) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt > theDeadline) return __LINE__;
  }
  {
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + theDelay;
    if(state.WaitNot(theDelay,0)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt < theDeadline) return __LINE__;

  }
  {
    RavlN::DateC theDeadline = RavlN::DateC::NowUTC() + 0.5;
    if(!state.WaitNot(theDelay,1)) return __LINE__;
    RavlN::DateC expiredAt = RavlN::DateC::NowUTC();
    if(expiredAt > theDeadline) return __LINE__;
  }

  std::cout << "ThreadState test passed. \n";
  return 0;
}
