
#import <Foundation/Foundation.h>

#include "Ravl/EntryPnt.hh"
#include "Ravl/Threads/LaunchThread.hh"

#define DPDEBUG 0
#if DPDEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  static int g_returnCode = 0;
  static volatile bool g_timeToExit = false;
  
  bool NewMainThread(int argc, char* argv[],int (*func)(int argc, char* argv[])) {
    NSAutoreleasePool *pool =[[NSAutoreleasePool alloc] init];
    g_returnCode = func(argc,argv);
    g_timeToExit = true;
    [pool release];
    return true;
  }
  
  int MacOSXMainCallManager(int argc, char* argv[],int (*func)(int argc, char* argv[])) {
    NSAutoreleasePool *pool =[[NSAutoreleasePool alloc] init];
    
    LaunchThread(&NewMainThread,argc,argv,func);
    
    ONDEBUG(std::cerr << "Starting MacOSX main loop \n");
    // FIXME:- Must be a way of signalling the run loop to exit??
    while(!g_timeToExit) {
      ONDEBUG(std::cerr << "Loop. \n");
      [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow: 1]];
    }
    ONDEBUG(std::cerr << "Completed MacOSX main loop \n");
    
    [pool release];
    return g_returnCode;
  }

  bool g_linkMacOSXRunLoop =  RegisterMainCallManager(&MacOSXMainCallManager);
  bool DoLinkMacOSXMainRunLoop()
  { return g_linkMacOSXRunLoop; }
}
