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
//! file = "Ravl/Contrib/EHS/HTTPServer.cc"

#include "Ravl/HTTPServer.hh"
#include "Ravl/EHS.hh"

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN
{
  
  
  
  static UIntT g_defThreadCount = 2;
  
  
  
  HTTPServerBodyC::HTTPServerBodyC(UIntT port) :
    m_port(port),
    m_ehs(NULL),
    m_sigHandle(HTTPRequestC(), HTTPResponseC(), EHTTPResponseCode_Invalid)
  {
  }
  
  
  
  HTTPServerBodyC::~HTTPServerBodyC()
  {
    if (m_ehs)
      delete m_ehs;
  }
  
  
  
  bool HTTPServerBodyC::Start()
  {
    RavlAssertMsg(m_ehs == NULL, "HTTPServerBodyC::Start existing server");
    
    // Build the parameters
    EHSServerParameters params;
    params["port"] = (IntT)m_port;
    params["mode"] = "threadpool";
    params["threadcount"] = (IntT)g_defThreadCount;
    
    // Create the server
    m_ehs = new RavlEHSC;
    Connect(m_ehs->SigHandle(), m_sigHandle);

    // Start the server
    m_ehs->StartServer(params);
    
    return true;
  }
  
  
  
  bool HTTPServerBodyC::Stop()
  {
    RavlAssertMsg(m_ehs != NULL, "HTTPServerBodyC::Stop no server");
    
    m_ehs->StopServer();
    
    return true;
  }
  
  
  
}

