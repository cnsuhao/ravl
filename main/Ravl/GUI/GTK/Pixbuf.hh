// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003,  OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLGUI_PIXBUF_HEADER
#define RAVLGUI_PIXBUF_HEADER 1
//! rcsid="$Id$"
//! lib=RavlGUI
//! author="Charles Galambos"
//! docentry="Ravl.GUI.Layout"
//! date="24/9/2003"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/GUI/Widget.hh"

#if RAVL_USE_GTK2

extern "C" {
  typedef struct _GdkPixbuf GdkPixbuf;
}

namespace RavlGUIN {
  using namespace RavlImageN;
  using namespace RavlN;
  
  //: Image buffer
  // GTK-2.x and above only.
  
  class PixbufC {
  public:
    PixbufC()
      : pixbuf(0)
    {}
    //: Default constructor.
    // Creates an invalid pix buf.
    
    PixbufC(const ImageC<ByteRGBValueC> &img);
    //: Create from an image.
    
    PixbufC(const char **data);
    //: Create from XPM data.
    
    PixbufC(const StringC &name);
    //: Create from filename
    
    PixbufC(GdkPixbuf *rawPixBuf);
    //: Handle raw pixbuf.
    
    ~PixbufC();
    //: Destructor.
    
    GdkPixbuf *Pixbuf()
    { return pixbuf; }

    const GdkPixbuf *Pixbuf() const
    { return pixbuf; }
    
    const IntT Rows() const;
    
    const IntT Cols() const;
    
  protected:
    GdkPixbuf *pixbuf;
  };
  
}
#else 
#warning PixbufC is not supported (requires gtk2)
#endif
#endif
