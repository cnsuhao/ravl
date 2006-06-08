// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/Assert.cc"

#include "Ravl/config.h"
#include "Ravl/Assert.hh"

#if RAVL_HAVE_ANSICPPHEADERS
#include <iostream>
#else
#include <iostream.h>
#endif

#if RAVL_HAVE_EXCEPTIONS
#include "Ravl/Exception.hh"
#endif

#include <stdio.h>

namespace RavlN {  
  using namespace std;

  static bool assertThrowException = false;
  
  //: Test if an exception on an assertion failure.
  
  bool AssertThrowException() {
#if RAVL_HAVE_EXCEPTIONS
    return assertThrowException;
#else
    return false;
#endif
  }
  
  //: Throw an exception on an assertion failure.
  // Note this may failed if exceptions are not supported.
  
  bool SetAssertThrowException(bool val) {
#if RAVL_HAVE_EXCEPTIONS
    assertThrowException = val;
    return true;
#else
    return val == false;
#endif
  }

  //: Called if assertion failed.
  
  void AssertFailed(char *file,int lineNo) {
    cerr << "Ravl assertion failed:" << file <<" " << lineNo << "\n";
#if RAVL_HAVE_EXCEPTIONS
    if(assertThrowException) 
      throw ExceptionAssertionFailedC("Ravl assertion failed. ");
#endif
    char *ptr = 0;
    *ptr = 1;// Cause a segfault.
  }
  
  //: Called if assertion failed, with message.
  
  void AssertFailed(char *file,int lineNo,char *msg) {
    cerr << "Ravl assertion failed:" << file <<" " << lineNo << "\n";
    cerr << "Reason: " << msg << "\n";
#if RAVL_HAVE_EXCEPTIONS
    if(assertThrowException)
      throw ExceptionAssertionFailedC(msg);
#endif
    char *ptr = 0;
    *ptr = 1;// Cause a segfault.
  }


  //: Called if assertion failed, with message.
  // This will either cause the program to trap to the debugger
  // or thow an 'ExceptionAssertionFailedC'
  
  void IssueError(char *file,int lineNo,char *msg ...) {
    const int formSize = 4096;
    va_list args;
    va_start(args,msg);
    char buff[formSize];
    int x = vsprintf(buff,msg,args);
    if(x > (formSize - 256)) 
      cerr << "OStreamC::form(...), WARNING: Ouput string is near or over buffer length. \n";
    cerr << buff;
    va_end(args);
#if RAVL_HAVE_EXCEPTIONS
    if(assertThrowException)
      throw ExceptionAssertionFailedC(buff,true);
#endif
    char *ptr = 0;
    *ptr = 1;// Cause a segfault.
  }

  //: Call when program encounters an a unexpected occurance.
  // this prints the message and continues.
  
  void IssueWarning(char *file,int lineNo,char *msg ...) {
    const int formSize = 4096;
    va_list args;
    va_start(args,msg);
    char buff[formSize];
    int x = vsprintf(buff,msg,args);
    if(x > (formSize - 256)) 
      cerr << "OStreamC::form(...), WARNING: Ouput string is near or over buffer length. \n";
    cerr << buff;
    va_end(args);
  }


}