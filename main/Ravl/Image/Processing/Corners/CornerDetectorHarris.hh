// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_CORNERHARRIS_HEADER
#define RAVLIMAGE_CORNERHARRIS_HEADER 1
//! rcsid="$Id$"
//! userlevel=Normal
//! date="1/10/1991"
//! lib=RavlImage
//! docentry="Ravl.Images.Corner Detection"
//! example=exCorner.cc
//! author="Radek Marik, modified by Charles Galambos"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/Corner.hh"
#include "Ravl/DList.hh"
#include "Ravl/Image/ConvolveSeparable2d.hh"

namespace RavlImageN {
  
  //! userlevel=Normal
  //: Harris corner detector.
  // Also known as the Plessey corner detector. Proposed by C.Harris in 1987.<p>
  // Note: the implementation of this detector could be faster.
  
  class CornerDetectorHarrisC {
  public:
    CornerDetectorHarrisC(int theshold = 20,int w = 5);
    //: Constructor.
    // threshold = Minimum level of cornerness to accept. <br>
    // w = width of filter to use for corners.
    
    DListC<CornerC> Apply(ImageC<ByteT> &img);
    //: Get a list of corners from 'img'
    
  protected:    
    void ImagGrad(ImageC<ByteT> &In,ImageC<TFVectorC<IntT,3> > &val);
    
    ImageC<IntT> CornerHarris(ImageC<ByteT> &img);
    
    int Peak(ImageC<IntT> &result,ImageC<ByteT> &in,DListC<CornerC> &cornerOut);

  private:
    const int w;
    const int threshold;
    ConvolveSeparable2dC<IntT,TFVectorC<IntT,3>,TFVectorC<IntT,3>,TFVectorC<IntT,3> > filter;    
    RealT maskSum;
    
    // Working images, so we don't have to repeated allocate them.
    ImageC<TFVectorC<IntT,3> > vals;
    ImageC<TFVectorC<IntT,3> > fvals;
    ImageC<IntT> var;
  };
  
}

#endif
