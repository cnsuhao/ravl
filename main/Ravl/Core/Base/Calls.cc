// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/Calls.cc"

#include "Ravl/Calls.hh"

namespace RavlN
{
#if !(RAVL_COMPILER_VISUAL_CPP || RAVL_COMPILER_MIPSPRO )
  template CallFunc0BodyC<bool>;  
#endif
  
  //: Invoke event.
  
  //void TriggerFunc0BodyC::Invoke()
  
  
};
