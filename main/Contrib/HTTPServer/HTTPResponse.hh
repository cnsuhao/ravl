// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_HTTPRESPONSE_HEADER
#define RAVL_HTTPRESPONSE_HEADER 1
//////////////////////////////////////////////////////////////////
//! rcsid = "$Id$"
//! lib = RavlEHS
//! author = "Warren Moore"
//! file = "Ravl/Contrib/EHS/HTTPResponse.hh"

#include "Ravl/String.hh"



class HttpResponse;
// Pre-declare the EHS response implementation



namespace RavlN
{
  
  
  
  /////////////////////////////
  //! userlevel = Normal
  //: HTTP Server Response
  
  class HTTPResponseC
  {
  public:
    HTTPResponseC();
    //: Default constructor.
    // Creates an invalid handle.
    
    HTTPResponseC(HttpResponse *response);
    //: Constructor
    
    bool SetText(StringC &text);
    //: Set the response text
    
  protected:
    HttpResponse *m_response;
    //: EHS pimpl
  };
  
  
  
}

#endif
