// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlMath

#include "Ravl/CircleIter.hh"


namespace RavlN {
  
  void CircleIterC::First() {
    x = 0;
    y = radius;
    d = 1 - radius;
    deltaE = 3;
    deltaSE = -2 * radius + 5;
    data = Index2dC(x,y) + offset;
    switch(radius)
      {
      case 0:
	x = y;
	octant = 7;
	break;
      case 1:
	octant = 4; 
	d += deltaSE;
	deltaE += 2;
	deltaSE += 4;
	y--; 
	x++;
	data = Index2dC(x,y) + offset;
	break;
      default:
	octant = 0;
      }
  }
    
  bool CircleIterC::Next() {
    octant++;
    switch(octant)
      {
      case 8: // Calculate next. octant 0
	octant = 0;
	if(y <= x) {
	  octant = -1;
	  return false; // Finished !
	}
	if(d < 0) {
	  d += deltaE;
	  deltaE += 2;
	  deltaSE += 2;
	} else {
	  d += deltaSE;
	  deltaE += 2;
	  deltaSE += 4;
	  y--;
	}
	x++;
	data = Index2dC(x,y) + offset;
	break;
	
      case 1:
	if(x != 0) {
	  data = Index2dC(-x,-y) + offset;
	  break;
	}
	octant++;
      case 2:
	data = Index2dC(y,-x) + offset;
	break;
      case 3:
	data = Index2dC(-y,x) + offset;
	if(x == y)
	  octant = 7; // Skip redundant cases.
	break;
	
	// Only done is x != y
      case 4:
	data = Index2dC(x,-y) + offset;
	if(x == 0)
	  octant = 7;
	break;
      case 5:
	data = Index2dC(y,x) + offset;
	break;
      case 6:
	data = Index2dC(-y,-x) + offset;
	break;      
      case 7:
	data = Index2dC(-x,y) + offset;
	break;
	
      default:
	RavlAssertMsg(0,"Bad state.");
	break;
      }
    return true;
  }
}

