// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_HTTPSERVER_HEADER
#define RAVL_HTTPSERVER_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlEHS
//! author = "Warren Moore"
//! file = "Ravl/Contrib/EHS/HTTPServer.hh"

#include "Ravl/RefCounter.hh"
#include "Ravl/String.hh"
#include "Ravl/Threads/Signal3.hh"
#include "Ravl/HTTPRequest.hh"
#include "Ravl/HTTPResponse.hh"

namespace RavlN
{
  
  
  
  class RavlEHSC;

  
  
  enum HTTPResponseCodeT
  {
    EHTTPResponseCode_Invalid = 0,
    EHTTPResponseCode_200_Ok = 200,
    EHTTPResponseCode_301_MovedPermanently = 301,
    EHTTPResponseCode_302_Found = 302,
    EHTTPResponseCode_401_Unauthorized = 401,
    EHTTPResponseCode_403_Forbidden = 403,
    EHTTPResponseCode_404_NotFound = 404,
    EHTTPResponseCode_500_InternalServerError = 500
  };
  //: Return code enumeration

  
  
  /////////////////////////////
  //! userlevel = Develop
  //: HTTP Server Body
  
  class HTTPServerBodyC :
    public RCBodyC
  {
  public:
    HTTPServerBodyC(UIntT port);
    //: Constructor
    
    ~HTTPServerBodyC();
    //: Destructor
    
    bool Start();
    //: Start the server
    
    bool Stop();
    //: Stop the server
    
    Signal3C< HTTPRequestC, HTTPResponseC, HTTPResponseCodeT > &SigHandle()
    { return m_sigHandle; }
    //: Get the signal handle
    
  protected:
    UIntT m_port;
    RavlEHSC *m_ehs;
    Signal3C< HTTPRequestC, HTTPResponseC, HTTPResponseCodeT > m_sigHandle;
  };

  
  
  /////////////////////////////
  //! userlevel = Normal
  //: HTTP Server
  // BIG OBJECT
  
  class HTTPServerC :
    public RCHandleC<HTTPServerBodyC>
  {
  public:
    HTTPServerC()
    {}
    //: Default constructor.
    // Creates an invalid handle.

    HTTPServerC(bool) :
       RCHandleC<HTTPServerBodyC>(*new HTTPServerBodyC(8080))
    {}
    //: Constructor.

    HTTPServerC(UIntT port, bool threaded) :
       RCHandleC<HTTPServerBodyC>(*new HTTPServerBodyC(port))
    {}
    //: Constructor.

    bool Start()
    { return Body().Start(); }
    //: Start the server

    bool Stop()
    { return Body().Stop(); }
    
    Signal3C< HTTPRequestC, HTTPResponseC, HTTPResponseCodeT > &SigHandle()
    { return Body().SigHandle(); }
    //: Get the signal handle
  };
  
  
  
}

#endif
