// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlVideoIO

//#include "Ravl/Image/VidIO.hh"
#include "Ravl/Image/ImgIO.hh"

namespace RavlImageN {

  //extern void InitDPImageIO();
  extern void InitRawIOFormat();
  extern void InitCifFormat();
  extern void InitYUVFormat();
  extern void InitRGBFormat();
  extern void InitSYUVFormat();

  void InitVidIO() {
    //InitDPImageIO();
    InitRawIOFormat();
    InitCifFormat();
    InitYUVFormat();
    InitRGBFormat();
    InitSYUVFormat();
    // InitImgIOComposites(); 
  }
}

