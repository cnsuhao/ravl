// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_RECTANGLEITER_HEADER
#define RAVLIMAGE_RECTANGLEITER_HEADER 1
//! rcsid="$Id$"
//! lib=RavlImageProc
//! author="Charles Galambos"
//! docentry="Ravl.Images.Misc"

#include "Ravl/Image/ImageRectangle.hh"

namespace RavlImageN {
  
  //! userlevel=Normal
  //: Rectangle iterator.
  // Slide a window over all possible positions in a
  // larger rectangle.
  
  class Rectange2dIterC {
  public:
    Rectange2dIterC(const ImageRectangleC &nImageRect,const ImageRectangleC &nwindow)
      : size2(nwindow.Range2().Size()),
	imageRect(nImageRect),
	window(nwindow)
    { First(); }
    //: Constructor.
    
    ImageRectangleC &Window()
    { return window; }
    //: Access current window.

    const ImageRectangleC &Window() const
    { return window; }
    //: Access current window.
    
    void First() { 
      UIntT size1 = window.Range1().Size();
      window.Range1().Min() = imageRect.Range1().Min();
      window.Range1().Max() = window.Range1().Min() + size1 - 1;
      window.Range2().Min() = imageRect.Range2().Min();
      window.Range2().Max() = window.Range2().Min() + size2 - 1;
    }
    //: Goto first position.
    
    bool Next() {
      ++(window.Range2());
      if(window.Range2().Max() <= imageRect.Range2().Max())
	return true;
      window.Range2().Min() = imageRect.Range2().Min();
      window.Range2().Max() = window.Range2().Min() + size2 - 1;
      ++(window.Range1());
      return false;
    }
    //:Goto next position.
    // Returns true if window is on the same row.

    bool operator++(int) 
    { return Next(); }
    //:Goto next position.
    // Returns true if window is on the same row.
    
    bool IsElm() const
    { return window.Range1().Max() <= imageRect.Range1().Max(); }
    //: At a valid position ?

    operator bool() const
    { return IsElm(); }
    //: At a valid position ?
    
  public:
    UIntT size2;
    ImageRectangleC imageRect;
    ImageRectangleC window;
  };
}


#endif
