// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_DEINTERLACE_HEADER
#define RAVLIMAGE_DEINTERLACE_HEADER 1
////////////////////////////////////////////////////////////////////////
//! date="30/1/2003"
//! author="Charles Galambos"
//! docentry="Ravl.API.Images.Video"
//! rcsid="$Id$"
//! lib=RavlImageProc
//! file="Ravl/Image/Processing/Filters/DeinterlaceStream.hh"

#include "Ravl/DP/StreamOp.hh"
#include "Ravl/DP/SPort.hh"
#include "Ravl/DP/SPortAttach.hh"
#include "Ravl/Average.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Array2dSqr311Iter3.hh"
#include "Ravl/CallMethods.hh"

namespace RavlImageN {

  //! userlevel=Develop
  //: De-interlace base class.
  
  class DeinterlaceStreamBaseC {
  public:
    DeinterlaceStreamBaseC(const DPSeekCtrlC &newSeekCtrl,bool evenFieldDom = false);
    //: Constructor.
    
    virtual ~DeinterlaceStreamBaseC();
    //: Destructor.
    
    bool HandleGetAttr(const StringC &attrName,bool &attrValue);
    //: Get bool attribute.
    
    bool HandleSetAttr(const StringC &attrName,const bool &attrValue);
    //: Set bool attribute.
    
    bool HandleGetAttr(const StringC &attrName,RealT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    bool HandleSetAttr(const StringC &attrName,const RealT &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
  protected:
    void RegisterAttribs(AttributeCtrlC &attrCtrl);
    //: Register attributes.
    
    DPSeekCtrlC seekCtrl;
    bool evenFieldDominant;
    IntT state; // Field number.
    DPIPortBaseC inputBase; 
  };
  
  //! userlevel=Develop
  //: De-interlace an incoming stream of images.
  
  template<class PixelT>
  class DeinterlaceStreamBodyC
    : public DPISPortBodyC<ImageC<PixelT> >,
      public DeinterlaceStreamBaseC
  {
  public:
    DeinterlaceStreamBodyC(DPIPortC<ImageC<PixelT> > &inPort,bool nEvenFieldDominant = true)
      : DeinterlaceStreamBaseC(DPSeekCtrlAttachC(inPort,true),nEvenFieldDominant),
	input(inPort)
    { 
      deinterlace = TriggerR(*this,&DeinterlaceStreamBodyC<PixelT>::Deinterlace,ImageC<PixelT>(),ImageC<PixelT>(),ImageC<PixelT>());
      inputBase = input;
      RegisterAttribs(inPort); 
    }
    //: Constructor.
    
    DeinterlaceStreamBodyC(DPISPortC<ImageC<PixelT> > &inPort,bool nEvenFieldDominant = true)
      : DeinterlaceStreamBaseC(DPSeekCtrlAttachC((const DPSeekCtrlC &) inPort),nEvenFieldDominant),
	input(inPort)
    { 
      deinterlace = TriggerR(*this,&DeinterlaceStreamBodyC<PixelT>::Deinterlace,ImageC<PixelT>(),ImageC<PixelT>(),ImageC<PixelT>());
      inputBase = input;
      RegisterAttribs(inPort); 
    }
    //: Constructor.
    
    virtual StringC OpName() const
    { return StringC("deinterlace"); }
    //: Op type name.
    
    virtual ImageC<PixelT> Get()  {
      ImageC<PixelT> ret;
      if(!Get(ret))
	throw DataNotReadyC("Failed to deinterlace frame. ");
      return ret;
    }
    //: Get next piece of data.
    // May block if not ready, or it could throw an 
    // DataNotReadyC exception.
    
    virtual bool Get(ImageC<PixelT> &buff) {
      ImageC<PixelT> img;
      //cerr << "State=" << state << " Tell=" << Tell() << "\n";
      switch(state) 
	{
	case 10:
	  cerr << "DeinterlaceStreamC, WARNING: Stream sync lost. State=" << state << ". Restarting. \n";
	  // Fall through.
	case 0:
	  state = 10; // Mark as error if we get interrupted by an exception.
	  if(!input.Get(img))
	    return false;
	  deinterlace(img,fields[0],fields[1]);
	  state = 1;
	  buff = fields[evenFieldDominant ? 1 : 0];
	  return true;
	case 1:
	  buff = fields[evenFieldDominant ? 0 : 1];
	  state = 0;
	  return true;
	case 11:
	  cerr << "DeinterlaceStreamC, WARNING: Stream sync lost. State=" << state << ". Restarting. \n";
	  // Fall through.
	case 2: // Load frame for second field.
 	  state = 11; // Mark as error if we get interrupted by an exception.
	  if(!input.Get(img))
	    return false;
	  deinterlace(img,fields[0],fields[1]);
	  buff = fields[evenFieldDominant ? 0 : 1];
	  state = 0;
	  return true;
	default: // Something's going really wrong.
	  cerr << "DeinterlaceStreamC, WARNING: Stream sync lost. State=" << state << ".  \n";
	  RavlAssert(0);
	}
      return false;
    }
    //: Try and get next piece of data.
    
    virtual bool Seek(UIntT off) {
      if(!seekCtrl.Seek(off/2))
	return false;
      state = (off % 2);
      if(state == 1)
	state = 2; // Make sure we load the appropriate frame.
      //cerr << "Seek to " << off << " Frame=" << (off/2) << " State=" << state << "\n";
      return true;
    }
    //: Seek to location in stream.
    // Returns false, if seek failed. 
    // if an error occurred (Seek returned False) then stream
    // position will not be changed.
    
    virtual bool DSeek(IntT off) {
      UIntT tmp = Tell();
      if(tmp == ((UIntT) -1))
	return false; // If tell doesn't work it'll fail.
      IntT at = tmp;
      //cerr << "DSeek by " << off << " " << at  << " State=" << state << "\n";
      // There may be slight more efficient ways of doing this, but
      // it will work for now.
      if(off < 0 && (-off) > at)
	return false;
      at += off;
      return Seek(at);
    }
    //: Delta Seek, goto location relative to the current one.
    // if an error occurred (DSeek returned False) then stream
    // position will not be changed.
    
    virtual UIntT Tell() const {
      UIntT at =seekCtrl.Tell(); 
      if(at == ((UIntT) -1))
	return (UIntT) -1;
      IntT fn = at * 2;
      switch(state) {
      case 2: fn += 1; break;
      case 1: fn -= 1; break;
      case 0:
      case 10: // Error.
      case 11: // Error.
	break;
      default: // Fatal error.
	RavlAssert(0);
      }
      return fn;
    }
    //: Find current location in stream.
    // Defined as the index of the next object to be written or read.
    // May return ((UIntT) (-1)) if not implemented.
    
    virtual UIntT Size() const { 
      UIntT size = seekCtrl.Size();
      if(size == ((UIntT) -1))
	return size;
      return size * 2; 
    }
    //: Find the total size of the stream. (assuming it starts from 0)
    // May return ((UIntT) (-1)) if not implemented.
    
    virtual UIntT Start() const
    { return seekCtrl.Start() * 2; }
    //: Find the offset where the stream begins, normally zero.
    // Defaults to 0

    virtual bool IsGetReady() const {
      if(!input.IsValid())
	return false;
      return input.IsGetReady(); 
    }
    //: Is some data ready ?
    // true = yes.
    
    virtual bool IsGetEOS() const {
      if(!input.IsValid())
	return true;
      return input.IsGetEOS(); 
    }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    virtual DPPortC ConnectedTo() const
    { return input; }
    //: What does this connect to ?
    
    virtual DListC<DPIPortBaseC> IPorts() const {
      DListC<DPIPortBaseC> lst;
      lst.InsLast(DPIPortBaseC((DPIPortBaseBodyC &)*this));
      return lst;
    }
    //: Input ports.
    
    virtual DListC<DPIPlugBaseC> IPlugs() const {
      DListC<DPIPlugBaseC> lst;
      lst.InsLast(DPIPlugC<ImageC<PixelT> >(input,"In1",DPEntityC((DPEntityBodyC &)*this)));
      return lst;
    }
    //: Input plugs.
    
    virtual bool GetAttr(const StringC &attrName,bool &attrValue) {
      if(HandleGetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::GetAttr(attrName,attrValue);
    }
    //: Get bool attribute.
    
    virtual bool SetAttr(const StringC &attrName,const bool &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::SetAttr(attrName,attrValue);
    }
    //: Set bool attribute.
    
    virtual bool GetAttr(const StringC &attrName,RealT &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const RealT &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
	return true;
      return AttributeCtrlBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool DeinterlaceFunc(const CallFunc3C<const ImageC<PixelT>&,ImageC<PixelT>&,ImageC<PixelT>& > &func)
    { deinterlace = func; return deinterlace.IsValid(); }
    //: Set de-interlacing function.
    
  protected:
    bool Deinterlace(const ImageC<PixelT> &img,ImageC<PixelT> &field0,ImageC<PixelT> &field1);
    //: DeinterlaceStream a frame.
    
    CallFunc3C<const ImageC<PixelT>&,ImageC<PixelT>&,ImageC<PixelT>& > deinterlace;
    ImageC<PixelT> fields[2];
    DPIPortC<ImageC<PixelT> > input; // Where to get data from.
  };
  
  
  //! userlevel=Normal
  //: Deinterlace an incoming stream of images.
  //
  // <p>The default behaviour of this class is to return fields rescaled to match the original frames size, as defined in the <code>DeinterlaceStreamBodyC<PixelT>::Deinterlace()</code> method (see the header file linked above).  If different behaviour is required, use the <code>DeinterlaceFunc()</code> method, as described below.</p>
  
  template<class PixelT>
  class DeinterlaceStreamC
    : public DPISPortC<ImageC<PixelT> >
  {
  public:
    DeinterlaceStreamC()
      : DPEntityC(true)
    {}
    //: Default constructor.
    // Creates an invalid handle.
    
    DeinterlaceStreamC(DPIPortC<ImageC<PixelT> > &inPort,bool nEvenFieldDominant = false)
      : DPEntityC(*new DeinterlaceStreamBodyC<PixelT>(inPort,nEvenFieldDominant))
    {}
    //: Constructor.
    
    DeinterlaceStreamC(DPISPortC<ImageC<PixelT> > &inPort,bool nEvenFieldDominant = false)
      : DPEntityC(*new DeinterlaceStreamBodyC<PixelT>(inPort,nEvenFieldDominant))
    {}
    //: Constructor.

  protected:
    DeinterlaceStreamBodyC<PixelT> &Body()
    { return dynamic_cast<DeinterlaceStreamBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.

    const DeinterlaceStreamBodyC<PixelT> &Body() const
    { return dynamic_cast<const DeinterlaceStreamBodyC<PixelT> &>(DPEntityC::Body()); }
    //: Access body.
    
  public:    
    bool DeinterlaceFunc(const CallFunc3C<const ImageC<PixelT> &,ImageC<PixelT> &,ImageC<PixelT> &> &func)
    { return Body().DeinterlaceFunc(func); }
    //: Set user-defined de-interlacing function.
    // The user-defined function <code>func</code> replaces the default
    // expansion of the field to a frame-sized image.  The function should take
    // 3 arguments, in this order: 
    // <blockquote>
    // <code>input frame</code><br>
    // <code>output field 1</code><br>
    // <code>output field 2</code><br>
    // </blockquote>
  };
  
  //: DeinterlaceStream a frame.
  
  template<class PixelT>
  bool DeinterlaceStreamBodyC<PixelT>::Deinterlace(const ImageC<PixelT> &img,ImageC<PixelT> &field0,ImageC<PixelT> &field1) {
    // Make sure images are allocated.
    field0 = ImageC<PixelT>(img.Frame());
    field1 = ImageC<PixelT>(img.Frame());
    
    for(Array2dSqr311Iter3C<PixelT,PixelT,PixelT> it(img,field0,field1);it;) {
      // Do even lines.
      do {
	it.Data3() = it.DataMM1();
	it.Data2() = Average<PixelT>(it.DataTM1(),it.DataBM1());
      } while(it.Next());
      if(!it)
	break;
      // Do odd lines.
      do {
	it.Data2() = it.DataMM1();
	it.Data3() = Average<PixelT>(it.DataTM1(),it.DataBM1());
      } while(it.Next());
      if(!it)
	break;
    }
    return true;
  }
  
  
}

#endif

