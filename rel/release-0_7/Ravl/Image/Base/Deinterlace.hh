// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_DEINTERLACE_HEADER
#define RAVLIMAGE_DEINTERLACE_HEADER 1
/////////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! author="Charles Galambos"
//! date="10/04/2002"
//! docentry="Ravl.Images"
//! lib=RavlImage
//! file="Ravl/Image/Base/Deinterlace.hh"

#include "Ravl/Image/Image.hh"
#include "Ravl/Array2dIter.hh"

namespace RavlImageN {
  
  template<class DataT>
  Array2dC<DataT> Deinterlace(Array2dC<DataT> &img,bool field1Dom = true) {
    RavlAssert((img.Range1().Size() % 2) == 0);
    Buffer2dC<DataT> newBuf(img.Buffer2d().Data(),img.Range2().Size());
    RangeBufferAccess2dC<DataT> rba(img.Range2());
    rba.Attach(newBuf,img.Range1());
    IndexC offset = img.Range1().Size() /2;
    IndexC end = offset + img.Range1().Min();    
    BufferAccessIterC<BufferAccessC<DataT> > it(rba);
    if(field1Dom) {
      for(IndexC i = img.Range1().Min();i < end;i++) {
	*it = img.RowPtr(i);
	it++;
	*it = img.RowPtr(i + offset);
	it++;
      }
    } else {
      for(IndexC i = img.Range1().Min();i < end;i++) {
	*it = img.RowPtr(i + offset);
	it++;
	*it = img.RowPtr(i);
	it++;
      }
    }
    return Array2dC<DataT>(rba,newBuf);
  }
  //: Deinterlace two fields held in a single frame.
  // This is an inplace operation and works by rearranging the row pointers
  // in the returned frame. This makes it very quick, but may cause problems
  // with functions that assume a simple linear memory layout for 2d arrays

  template<class DataT>
  Array2dC<DataT> DeinterlaceSubsample(Array2dC<DataT> &img,bool field1Dom = true) {
    RavlAssert((img.Range1().Size() % 2) == 0);
    Array2dC<DataT> newBuf(img.Range1().Size()/2, img.Range2().Size()/2);
    for(Array2dIterC<DataT>It(newBuf);It.IsElm();It.Next()) {
      It.Data() = img[It.Index().Row()*2][It.Index().Col()*2];
    }
    return newBuf;
  }
  //: Deinterlace and subsample, i.e. returns image that is half the size
  
}

#endif
