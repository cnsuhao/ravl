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
//! file = "Ravl/Contrib/EHS/HTTPResponse.cc"

#include "Ravl/HTTPResponse.hh"

#include <ehs.h>

#define DODEBUG 0

#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN
{
  
  
  
  HTTPResponseC::HTTPResponseC() :
    m_response(NULL)
  {
  }


  
  HTTPResponseC::HTTPResponseC(HttpResponse *response) :
    m_response(response)
  {
  }
  
  
  
  bool HTTPResponseC::SetText(StringC &text)
  {
    RavlAssertMsg(m_response, "HTTPResponseC::SetText no valid response");
    
    m_response->SetBody(text.chars(), text.length());
    
    return true;
  }
  
  
  
}

