// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/Exception.cc"

#include "Ravl/Exception.hh"

#if RAVL_HAVE_ANSICPPHEADERS
#include <iostream>
#include <typeinfo>
#else
#include <iostream.h>
#include <typeinfo.h>
#endif

#ifdef __GLIBC__
#include <execinfo.h>
#endif

namespace RavlN {

  //: Constructor
  ExceptionC::ExceptionC(const char *ntext,StackTraceT stackTrace)
   : desc(ntext),
     ref(false),
     m_stackTraceDepth(0)
  {
#ifdef __GLIBC__
    if(stackTrace == StackTraceOn) {
      m_stackTraceDepth = backtrace(m_stackTrace, m_maxStackDepth);
    }
#endif
  }

  //: Copy Constructor
  ExceptionC::ExceptionC(const ExceptionC &oth)
   : desc(oth.desc),
     ref(oth.ref),
     m_stackTraceDepth(0)
  {
    if(oth.ref)
      const_cast<ExceptionC &>(oth).ref = false;
  }

  //: Constructor
  
  ExceptionC::ExceptionC(const char *ntext,bool copy,StackTraceT stackTrace)
    : desc(ntext),
      ref(copy)
  {
#ifdef __GLIBC__
    if(stackTrace == StackTraceOn) {
      m_stackTraceDepth = backtrace(m_stackTrace, m_maxStackDepth);
    }
#endif

    //cerr << "String: '" << ntext << "'\n";
    if(!copy)      
      return;
    char *place = const_cast<char *>(ntext);
    // How long is the string ?
    int len = 1;
    for(;*place != 0;place++,len++) ;
    char *str = new char[len];
    // Copy it.
    desc = str;
    for(place = str;*ntext != 0;place++,ntext++) 
      *place = *ntext;
    *place = 0;
    //  cerr << "Stored: '" << desc << "'\n";
  }
  
  //: Virtualise destructor.
  
  ExceptionC::~ExceptionC()  {
    if(ref)
      delete [] desc;
  }
  
  //: Dump contents of exception to cerr;
  
  void ExceptionC::Dump(ostream &out) {
    out << "Ravl Exception:" << desc << endl;
#ifdef __GLIBC__
    char **trace = backtrace_symbols(m_stackTrace,m_stackTraceDepth);
    if(trace != 0) {
      out << "Stack:\n";
      for(int i = 0;i < m_stackTraceDepth;i++) {
        out << "  " << trace[i] << "\n";
      }
      free(trace);
    } else {
      out << "Stack: Failed to generate trace.";
    }
#endif
  }


  //: Dump contents of exception to cerr;
  
  void ExceptionErrorCastC::Dump() {
    cerr << "RAVL Exception: Illegal cast. \n";
    cerr << " From:" << from.name() << endl;
    cerr << "   To:" << to.name() << endl;
    cerr << " Desc:" << desc << endl << endl;
  }

  // ---------------------------------------------------------

  ExceptionErrorCastC::ExceptionErrorCastC(const char *ndesc,const type_info &nfrom,const type_info &nto)
   : ExceptionC(ndesc),
     from(nfrom),
     to(nto)
  {}

  // ---------------------------------------------------------

  ExceptionOperationFailedC::ExceptionOperationFailedC(const char *ndesc)
    : ExceptionC(ndesc)
  {}
    //: Constructor.

  ExceptionOperationFailedC::ExceptionOperationFailedC(const char *ntext,bool copy)
    : ExceptionC(ntext,copy)
  {}
  //: Constructor.
  // if copy is true, a copy is made of string ntext.

  // ---------------------------------------------------------

  ExceptionBadConfigC::ExceptionBadConfigC(const char *ndesc)
   : ExceptionC(ndesc)
   {}
  //: Constructor.
  
  ExceptionBadConfigC::ExceptionBadConfigC(const char *ntext,bool copy)
   : ExceptionC(ntext,copy)
  {}
  //: Constructor.
  // if copy is true, a copy is made of string ntext.

  // ---------------------------------------------------------
  //! Constructor.

  ExceptionUnexpectedVersionInStreamC::ExceptionUnexpectedVersionInStreamC(const char *text)
    : ExceptionC(text)
  {}

  // ---------------------------------------------------------

  //: Constructor.
  ExceptionOutOfMemoryC::ExceptionOutOfMemoryC(const char *ndesc)
    : ExceptionC(ndesc)
  {}

  //: Constructor.
  // if copy is true, a copy is made of string ntext.
  ExceptionOutOfMemoryC::ExceptionOutOfMemoryC(const char *ntext,bool copy)
   : ExceptionC(ntext,copy)
  {}

  // ---------------------------------------------------------

  //: Constructor.
  ExceptionOutOfRangeC::ExceptionOutOfRangeC(const char *ndesc)
    : ExceptionC(ndesc)
  {}

  //: Constructor.
  // if copy is true, a copy is made of string ntext.
  ExceptionOutOfRangeC::ExceptionOutOfRangeC(const char *ntext,bool copy)
    : ExceptionC(ntext,copy)
  {}

  // ---------------------------------------------------------

  //: Constructor.
  ExceptionAssertionFailedC::ExceptionAssertionFailedC(const char *ndesc)
   : ExceptionC(ndesc)
  {}

  //: Constructor.
  // if copy is true, a copy is made of string ntext.
  ExceptionAssertionFailedC::ExceptionAssertionFailedC(const char *ntext,bool copy)
   : ExceptionC(ntext,copy)
  {}

  // ---------------------------------------------------------

  //: Constructor.
  ExceptionNumericalC::ExceptionNumericalC(const char *ndesc)
   : ExceptionC(ndesc)
  {}

  //: Constructor.
  // if copy is true, a copy is made of string ntext.
  ExceptionNumericalC::ExceptionNumericalC(const char *ntext,bool copy)
   : ExceptionC(ntext,copy)
  {}

  // ---------------------------------------------------------

  //: Constructor
  ExceptionEndOfStreamC::ExceptionEndOfStreamC(const char *ntext)
   : ExceptionC(ntext)
  {}

  //: Constructor.
  // if copy is true, make a copy of ntext.
  ExceptionEndOfStreamC::ExceptionEndOfStreamC(const char *ntext,bool copy)
   : ExceptionC(ntext,copy)
  {}

}
