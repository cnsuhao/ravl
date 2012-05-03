// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2004-12, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef EX_HH
#define EX_HH
//! lib=RavlClassWizard

#include "Ravl/RefCounter.hh"
#include "Ravl/Stream.hh"

using namespace RavlN;

//! userlevel=Develop
//: Example body class.

class AnObjectBodyC
  : public RCBodyC
{
public:
  AnObjectBodyC(int a)
    : someData(a)
  {}
  //: Constructor.
  
  int Data()
  { return someData; }
  //: Access data.
  
  void SetData(int x)
  { someData = x; }
  //: Set member data.
  
protected:
  int someData;
};
  
    
  
#endif

