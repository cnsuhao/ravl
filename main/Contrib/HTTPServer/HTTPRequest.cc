// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlEHS
//! author = "Warren Moore"
//! file = "Ravl/Contrib/EHS/HTTPRequest.cc"

#include "Ravl/HTTPRequest.hh"
#include "Ravl/Assert.hh"

#include <iostream>
#include <string>
#include <ehs.h>

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN
{
  
  
  
  HTTPRequestC::HTTPRequestC() :
    m_request(NULL),
    m_method(EHTTPRequest_INVALID)
  {
  }


  
  HTTPRequestC::HTTPRequestC(HttpRequest *request) :
    m_request(request),
    m_method(EHTTPRequest_INVALID)
  {
    RavlAssertMsg(m_request != NULL, "HTTPRequestC::HTTPRequestC invalid request object");
    
    switch (m_request->nRequestMethod)
    {
      case REQUESTMETHOD_GET:
        m_method = EHTTPRequest_GET;
        break;
      case REQUESTMETHOD_POST:
        m_method = EHTTPRequest_POST;
        break;
      case REQUESTMETHOD_UNKNOWN:
      case REQUESTMETHOD_INVALID:
      default:
        m_method = EHTTPRequest_UNKNOWN;
        RavlAssertMsg(false, "HTTPRequestC::HTTPRequestC unknown request method");
    }
  }


    
  StringC HTTPRequestC::URI()
  {
    RavlAssertMsg(m_request != NULL, "HTTPRequestC::URI invalid request object");
    RavlAssertMsg(m_method != EHTTPRequest_UNKNOWN && \
                  m_method != EHTTPRequest_INVALID, "HTTPRequestC::URI invalid request method");
                  
    return StringC(m_request->sUri.c_str());
  }
  
  
  
  StringC HTTPRequestC::OriginalURI()
  {
    RavlAssertMsg(m_request != NULL, "HTTPRequestC::OriginalURI invalid request object");
    RavlAssertMsg(m_method != EHTTPRequest_UNKNOWN && \
                  m_method != EHTTPRequest_INVALID, "HTTPRequestC::OriginalURI invalid request method");
                  
    return StringC(m_request->sOriginalUri.c_str());
  }
  
  
  
}

