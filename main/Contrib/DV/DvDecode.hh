// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_DVDECODE_HEADER
#define RAVLIMAGE_DVDECODE_HEADER 1
////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/Contrib/DV/DvDecode.hh"
//! lib=RavlDV
//! author="Kieron Messer"
//! docentry="Ravl.Contrib.Video IO.DV"
//! date="15/05/2002"

#include "Ravl/DP/Process.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Array1d.hh"
#include "libdv/dv.h"

namespace RavlImageN {

  using namespace RavlN;
  
  // --------------------------------------------------------------------------
  // **********  DvDecodeBodyC  ********************************************
  // --------------------------------------------------------------------------
  //! userlevel=Develop
  //: 
  // You should not use this class directly, but rather the handle class
  // DvDecode.
  //
  
  
  class DvDecodeBodyC
    : public DPProcessBodyC<ByteT *, ImageC<ByteRGBValueC> >
  {
  
  public:
    DvDecodeBodyC (bool deinterlace);
    //: Constructor
    // (See handle class DvDecode)
    
    ~DvDecodeBodyC();
    //: Destructor.
    
    ImageC<ByteRGBValueC> Apply (ByteT * arr);
    //: Converts PAL DV frame to byte RGB image
    
    virtual bool IsStateless() const 
    { return false; }
    //: Is operation stateless ?
    
    bool Deinterlace() const
    { return deinterlace; }
    //: Are we deinterlaceing ?
    
    bool Deinterlace(bool val)
    { return deinterlace = val; }
    //: Set deinterlace flag.
    
  private:
    dv_decoder_t *decoder;
    uint8_t *decoded;
 
    bool init;
    bool deinterlace;
  };
  
  ///////////////////////////////////////////////////
  
  
  // --------------------------------------------------------------------------
  // **********  DvDecode  ********************************************
  // --------------------------------------------------------------------------
  
  //! userlevel=Normal
  //: Function to decode raw DV into a RAVL image
  
  class DvDecodeC
    : public DPProcessC<ByteT *, ImageC<ByteRGBValueC> >
  {
  public:  
    DvDecodeC (bool deinterlace=false)
      : DPProcessC<ByteT *, ImageC<ByteRGBValueC> >(*new DvDecodeBodyC(deinterlace))
    {}
    //: Constructs DvDecodeC 

  protected:
    DvDecodeBodyC &Body() 
    { return static_cast<DvDecodeBodyC &>(DPEntityC::Body()); }
    //: Body access.
    
    const DvDecodeBodyC &Body() const
    { return static_cast<const DvDecodeBodyC &>(DPEntityC::Body()); }
    //: Body access.

  public:
    ImageC<ByteRGBValueC> Apply (ByteT * arr)
    { return Body().Apply(arr); }
    //: Converts PAL DV frame to byte RGB image
    
    bool Deinterlace() const
    { return Body().Deinterlace(); }
    //: Are we deinterlaceing ?
    
    bool Deinterlace(bool val)
    { return Body().Deinterlace(val); }
    //: Set deinterlace flag.
    
  };
  
} // end namespace RavlImageN

#endif
