// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef DPPORT_HEADER
#define DPPORT_HEADER 1
////////////////////////////////////////////////////
//! docentry="Ravl.Core.Data Processing" 
//! rcsid="$Id$"
//! file="Ravl/Core/IO/Port.hh"
//! lib=RavlIO
//! author="Charles Galambos"
//! date="16/06/98"
//! userlevel=Default

#include "Ravl/DP/Entity.hh"
#include "Ravl/Assert.hh"
#include "Ravl/Exception.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/SArr1Iter.hh"

#include <iostream.h>

#if RAVL_HAVE_ANSICPPHEADERS
#include <typeinfo>
#else
#include <typeinfo.h>
#endif 

namespace RavlN {
  class DPPortC;
  class StringC;
  
  //! userlevel=Normal
  //: Exception, Data Not Ready.
  // This is throw if a Get is unabled to
  // comlete because there is no data available.
  
  class DataNotReadyC : public ExceptionC {
  public:
    DataNotReadyC(const char *msg = "")
      : ExceptionC(msg)
      {}
  };
  
  //! userlevel=Develop
  //: Abstract port body.
  
  class DPPortBodyC 
    : virtual public DPEntityBodyC 
  {
  public:
    DPPortBodyC() {}
    //: Default constructor.
    
    DPPortBodyC(istream &in) 
      : DPEntityBodyC(in)
      {}
    //: Stream constructor.
    
    virtual ~DPPortBodyC() {}
    //: Destructor.
    
    virtual bool IsAsync() const;
    //: Does port work asynchronously ?
    
    virtual bool Save(ostream &out) const 
      { return DPEntityBodyC::Save(out); }
    //: Save to ostream.
    
    virtual DPPortC ConnectedTo() const;
    //: Is this port connected to another ?
    // If not returns invalid handle.
    
    virtual bool GetAttr(const StringC &attrName,StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    virtual bool SetAttr(const StringC &attrName,const StringC &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
  };
  
  //! userlevel=Develop
  //: Input port base body.
  
  class DPIPortBaseBodyC 
    : public DPPortBodyC 
  {
  public:
    DPIPortBaseBodyC() {}
    //: Default constuctor.
    
    DPIPortBaseBodyC(istream &in) 
      : DPPortBodyC(in)
      {}
    // Stream constuctor.
    
    virtual bool IsGetReady() const;
    //: Is some data ready ?
    // true = yes.
    // Defaults to !IsGetEOS().
    
    virtual bool IsGetEOS() const;
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    virtual const type_info &InputType() const;
    //: Input type.
    
    virtual bool Save(ostream &out) const 
      { return DPPortBodyC::Save(out); }
    //: Save to ostream.
    
    virtual bool Discard();
    //: Discard the next input datum.
  };
  
  //! userlevel=Develop
  //: Input port body.
  
  template<class DataT>
  class DPIPortBodyC 
    : public DPIPortBaseBodyC 
  {
  public:
    DPIPortBodyC() 
      {}
    //: Default constructor.
    
    DPIPortBodyC(istream &in) 
      : DPIPortBaseBodyC(in)
      {}
    //: Stream constructor.
    
    virtual DataT Get()  {
      cerr << "DPIPortBodyC<DataT>::Get(), ERROR: Abstract method called. \n";
      assert(0);
      return DataT();
    }
    //: Get next piece of data.
    // May block if not ready, or it could throw an 
    // DataNotReadyC exception.
    // NB. This function MUST be provided by client class.
    
    virtual bool Get(DataT &buff) { 
      try {
	buff = Get();
      } catch(DataNotReadyC &) {
	return false;
      }
      return true;
    }
    //: Try and get next piece of data.
    // This may not NOT block, if no data is ready
    // it will return false, and not set buff.
    // NB. The default version of this function uses
    // the Get() method defined above and so need 
    // not be provided by derived classes.
    
    virtual IntT GetArray(SArray1dC<DataT> &data);
    //: Get an array of data from stream.
    // returns the number of elements succesfully processed.
    // NB. This need NOT be overridden in client classes 
    // unless fast handling of arrays of data elements is required.
    
    virtual const type_info &InputType() const { return typeid(DataT); }
    // Input type.  
    
    virtual bool Save(ostream &out) const 
      { return DPIPortBaseBodyC::Save(out); }
    //: Save to ostream.
    
    virtual bool Discard() { 
      DataT tmp;
      return Get(tmp);
    }
    //: Discard the next input datum.
    
  };
  
  template<class DataT>
  IntT DPIPortBodyC<DataT>::GetArray(SArray1dC<DataT> &data) {
    for(SArray1dIterC<DataT> it(data);it;it++) {
      if(!Get(*it))
	return it.Index().V();
    }
    return data.Size();
  }
  
  //! userlevel=Develop
  //: Output port base body.
  
  class DPOPortBaseBodyC 
    : public DPPortBodyC 
  {
  public:
    DPOPortBaseBodyC() {}
    //: Default constuctor.
    
    DPOPortBaseBodyC(istream &in)
      : DPPortBodyC(in)
      {}
    //: Stream constuctor.
    
    virtual void PutEOS();
    //: Put End Of Stream marker.
    
    virtual bool IsPutReady() const;
    //: Is port ready for data ?
    
    virtual const type_info &OutputType() const;
    //: Output type.
    
    virtual bool Save(ostream &out) const 
      { return DPPortBodyC::Save(out); }
    //: Save to ostream.
  };
  
  //! userlevel=Develop
  //: Output port body.
  
  template<class DataT>
  class DPOPortBodyC 
    : public DPOPortBaseBodyC 
  {
  public:
    DPOPortBodyC() {}
    //: Default constructor.
    
    DPOPortBodyC(istream &in)
      : DPOPortBaseBodyC(in)
      {}
    //: Default constructor.
    
    virtual bool Put(const DataT &) { 
      cerr << "DPOPortBodyC<DataT>::Put(), ERROR: Abstract method called. \n";
      assert(0);
      return false; 
    }
    //: Put data.
    // This function MUST be provided by client class.
    
    virtual IntT PutArray(const SArray1dC<DataT> &data);
    //: Put an array of data to stream.
    // returns number of elements processed.
    // NB. This need NOT be overridden in client classes 
    // unless fast handling of arrays of data elements is required.
    
    virtual const type_info &OutputType() const { return typeid(DataT); }
    //: Input type.
    
    virtual bool Save(ostream &out) const 
      { return DPOPortBaseBodyC::Save(out); }
    //: Save to ostream.
  };
  
  template<class DataT>
  IntT DPOPortBodyC<DataT>::PutArray(const SArray1dC<DataT> &data) {
    for(SArray1dIterC<DataT> it(data);it;it++) {
      if(!Put(*it))
	return it.Index().V();
    }
    return data.Size();
  }
  
  //! userlevel=Develop
  //: Input/Output port body.
  
  template<class InT,class OutT>
  class DPIOPortBodyC 
    : public DPIPortBodyC<InT>, 
      public DPOPortBodyC<OutT> 
  {
  public:
    DPIOPortBodyC() {}
    //: Default constructor.
    
    DPIOPortBodyC(istream &in) 
      : DPIPortBodyC<InT>(in),
      DPOPortBodyC<OutT>(in)
      {}
    //: Stream constructor.
    
    virtual bool Save(ostream &out) const  {
      if(!DPIPortBodyC<InT>::Save(out))
	return false;
      return DPOPortBodyC<OutT>::Save(out);
    }
    //: Save to ostream.
    
  };
  
  template<class DataT> class DPPlugC;
  
  ///////////////////////////
  //! userlevel=Develop
  //: Base port handle.
  
  class DPPortC : virtual public DPEntityC {
  public:
    DPPortC() 
      : DPEntityC(false)
      {}
    //: Default constructor.
    
    DPPortC(DPPortBodyC &bod)
      : DPEntityC(bod)
      {}
    //: Constructor.
    
    DPPortC(istream &in)
      : DPEntityC(in)
      {}
    //: Stream constructor.
    
    DPPortC(const DPPortC &oth)
      : DPEntityC(oth)
  {}
    //: Copy constructor.
    
  protected:
    inline DPPortBodyC &Body() 
      { return dynamic_cast<DPPortBodyC &> (DPEntityC::Body()); }
    //: Access body.
    
    inline const DPPortBodyC &Body() const
      { return dynamic_cast<const DPPortBodyC &> (DPEntityC::Body()); }
    //: Access body.
    
  public:
    
    inline bool IsAsync() const 
      { return Body().IsAsync(); }
    //: Does port work asynchronously ??
    
    inline DPPortC ConnectedTo() const
      { return Body().ConnectedTo(); }
    //: Is this port connected to another ?
    // If not returns invalid handle.
    
    inline bool GetAttr(const StringC &attrName,StringC &attrValue)
      { return Body().GetAttr(attrName,attrValue); }    
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    StringC GetAttr(const StringC &attrName);
    //: Get a stream attribute.
    // Return the value of an attribute or an empty string if its unkown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
    inline bool SetAttr(const StringC &attrName,const StringC &attrValue)
      { return Body().SetAttr(attrName,attrValue); }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.
    
  };
  
  //////////////////////////
  //! userlevel=Develop
  //: Input port base class.
  
  class DPIPortBaseC 
    : public DPPortC 
  {
  public:
    DPIPortBaseC() 
      : DPEntityC(true)
      {}
    //: Default constructor.
    
    DPIPortBaseC(const DPIPortBaseC &oth) 
      : DPEntityC(oth),
      DPPortC(oth)
      {}
    //: Copy constructor.
    
    DPIPortBaseC(DPIPortBaseBodyC &bod) 
      : DPEntityC(bod),
      DPPortC(bod)
      {}
  //: Body constructor.
    
    DPIPortBaseC(istream &strm) 
      : DPEntityC(strm)
      {}
    //: Stream constructor.
    
    DPIPortBaseC(const DPPortC &bod) 
      : DPEntityC(bod),
      DPPortC(bod)
      {
	if(dynamic_cast<DPIPortBaseBodyC *>(&DPEntityC::Body()) == 0)
	  Invalidate();
      }
    //: Body constructor.
    
  protected:
    inline DPIPortBaseBodyC &Body() 
      { return dynamic_cast<DPIPortBaseBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
    inline const DPIPortBaseBodyC &Body() const
      { return dynamic_cast<const DPIPortBaseBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
  public:
    inline const type_info &InputType() const 
      { return Body().InputType(); }
    // Get type of input port.
    
    inline bool IsGetReady() const 
      { return Body().IsGetReady(); }
    // Is valid data.
    
    inline bool IsGetEOS() const 
      { return Body().IsGetEOS(); }
    // Is valid data.
  
    inline bool Discard()
    { return Body().Discard(); }
    //: Discard the next input datum.
    // returns true on success.
  };
  
  /////////////////////////////////
  //! userlevel=Normal
  //: Input port.
  
  template<class DataT>
  class DPIPortC 
    : public DPIPortBaseC 
  {
  public:
    DPIPortC() 
      : DPEntityC(true)
      {}
    //: Default constructor.
    
#ifdef __sgi__
    DPIPortC(const DPIPortC<DataT> &oth) 
      : DPEntityC(oth),
	DPIPortBaseC(oth)
      {}
    //: Copy constructor.
#endif
    
    DPIPortC(DPIPortBodyC<DataT> &bod) 
      : DPEntityC(bod),
	DPIPortBaseC(bod)
      {}
    //: Body constructor.
    
    DPIPortC(const DPIPortBaseC &oth) 
      : DPEntityC(oth)
    { 
#if RAVL_CHECK
      if(IsValid()) {
	if(oth.InputType() != typeid(DataT)) {
	  cerr << "DPIPortC<DataT>() Type mismatch.  " << oth.InputType().name() << " given to " << typeid(DataT).name() << endl; 
	  assert(0);
	  }
      }
#endif
    }
  //: Base constructor.
    
    DPIPortC(istream &in) 
      : DPEntityC(in)
      {}
    //: Stream constructor.
    
  protected:
    DPIPortBodyC<DataT> &Body() 
      { return dynamic_cast<DPIPortBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPIPortBodyC<DataT> &Body() const
      { return dynamic_cast<const DPIPortBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
  public:
    
    inline DataT Get() 
      { return Body().Get(); }
    // Get next piece of data.
    
    inline bool Get(DataT &buff) 
      { return Body().Get(buff); }
    //: Try and get next piece of data.
    // If none, return false.
    // else put data into buff.  
    
    inline IntT GetArray(SArray1dC<DataT> &data)
      { return Body().GetArray(data); }
    //: Get an array of data from stream.
    // returns the number of elements  processed.
    
    friend class DPPlugC<DataT>;
  };
  
  template <class DataT>
  ostream & operator<<(ostream & s,const DPIPortC<DataT> &port) { 
    port.Save(s); 
    return s; 
  }
  
  template <class DataT>
  istream & operator>>(istream & s, DPIPortC<DataT> &port) { 
    DPIPortC<DataT> nport(s); port = nport; 
    return s; 
  }
  
  ////////////////////////////////
  //! userlevel=Develop
  //: Output port base.
  
  class DPOPortBaseC 
    : public DPPortC 
  {
  public:
    DPOPortBaseC() 
      : DPEntityC(true)
      {}
    //: Default constructor.
    
    DPOPortBaseC(const DPOPortBaseC &oth) 
      : DPEntityC(oth),
      DPPortC(oth)
      {}
    // Copy constructor.
    
    DPOPortBaseC(DPOPortBaseBodyC &bod) 
      : DPEntityC(bod),
      DPPortC(bod)
      {}
    // Body constructor.
    
    DPOPortBaseC(istream &strm) 
      : DPEntityC(strm)
      {}
    // Stream constructor.
    
  protected:
    inline DPOPortBaseBodyC &Body() 
      { return dynamic_cast<DPOPortBaseBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
    inline const DPOPortBaseBodyC &Body() const
      { return dynamic_cast<const DPOPortBaseBodyC &>(DPEntityC::Body()); }
    //: Access body.
    
  public:
    inline void PutEOS() { Body().PutEOS(); }
    //: Put End Of Stream marker.
    
    inline const type_info &OutputType() const { return Body().OutputType(); }
    //: Get type of output port.
  };
  
  //////////////////////////////
  //! userlevel=Normal
  //: Output port.
  
  template<class DataT>
  class DPOPortC 
    : public DPOPortBaseC 
  {
  public:
    DPOPortC() 
      : DPEntityC(true)
      {}
    // Default constructor.
    
#ifdef __sgi__
    DPOPortC(const DPOPortC<DataT> &oth) 
      : DPEntityC(oth),
	DPOPortBaseC(oth)
      {}
    //: Copy constructor.
#endif
    
    DPOPortC(DPOPortBodyC<DataT> &bod) 
      : DPEntityC(bod),
	DPOPortBaseC(bod)
      {}
    //: Body constructor.
    
    DPOPortC(istream &in) 
      : DPEntityC(in)
      {}
    //: Stream constructor.
    
    DPOPortC(const DPOPortBaseC &oth) 
      : DPEntityC(oth)
      { 
#if RAVL_CHECK
	if(IsValid()) {
	  if(oth.OutputType() != typeid(DataT)) {
	    cerr << "DPOPortC<DataT>() Type mismatch.  " << oth.OutputType().name() << " given to " << typeid(DataT).name() << endl; 
	    assert(0);
	  }
	}
#endif
      }
    //: Base constructor.
    
  protected:
    DPOPortBodyC<DataT> &Body() 
      { return dynamic_cast<DPOPortBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPOPortBodyC<DataT> &Body() const
      { return dynamic_cast<const DPOPortBodyC<DataT> &>(DPEntityC::Body()); }
    //: Access body.
    
  public:
    inline bool Put(const DataT &dat) 
      { return Body().Put(dat); }
    //: Put data to stream
    // Returns true if data is put into stream succesfully.
    
    inline IntT PutArray(const SArray1dC<DataT> &data)
      { return Body().PutArray(data); }
    //: Put an array of data elements into a stream.
    // returns the number of elements processed from the array.
    
    inline bool IsPutReady() const 
      { return Body().IsPutReady(); }
    // Is port ready for data ?  
  };
  
  template <class DataT>
  ostream & operator<<(ostream & s,const DPOPortC<DataT> &port) { 
    port.Save(s); 
    return s; 
  }
  
  template <class DataT>
  istream & operator>>(istream & s, DPOPortC<DataT> &port) { 
    DPOPortC<DataT> nport(s); port = nport; 
    return s; 
  }
  
  //////////////////////////////
  //! userlevel=Normal
  //: Input/Output port.
  
  template<class InT,class OutT>
  class DPIOPortC 
    : public DPIPortC<InT>, 
      public DPOPortC<OutT> 
  {
  public:
    DPIOPortC() 
      : DPEntityC(true)
      {}
    // Default constructor.
    
    DPIOPortC(const DPIOPortC<InT,OutT> &oth) 
      : DPEntityC(oth),
      DPIPortC<InT>(oth),
      DPOPortC<OutT>(oth)
      {}
    //: Copy constructor.
    
    DPIOPortC(DPIOPortBodyC<InT,OutT> &bod) 
    : DPEntityC(bod),
      DPIPortC<InT>(bod),
      DPOPortC<OutT>(bod)
      {}
  //: Body constructor.
    
    DPIOPortC(istream &in) 
      : DPEntityC(in)
      {}
    //: Stream constructor.
    
    inline const DPIOPortC<InT,OutT> Copy() const  { 
      if(!IsValid())
	return DPIOPortC<InT,OutT>(); // Nothing to copy.
      return DPIOPortC<InT,OutT>(dynamic_cast<DPIOPortBodyC<InT,OutT> &>(Body().Copy())); 
    }
    //: Make a copy of this process.
    
    DPIPortC<InT> &In() { return *this; }
    //: Use as input port.
    // (Get from.)
    
    DPOPortC<OutT> &Out() { return *this; }
    //: Use as output port.
    // (Put to.)
    
  protected:
    DPIOPortBodyC<InT,OutT> &Body() 
      { return dynamic_cast<DPIOPortBodyC<InT,OutT> &>(DPEntityC::Body()); }
    //: Access body.
    
    const DPIOPortBodyC<InT,OutT> &Body() const
      { return dynamic_cast<const DPIOPortBodyC<InT,OutT> &>(DPEntityC::Body()); }
    //: Access body.
  };
  
  template <class InT,class OutT>
  ostream & operator<<(ostream & s,const DPIOPortC<InT,OutT> &port) { 
    port.Save(s); 
    return s; 
  }
  
  template <class InT,class OutT>
  istream & operator>>(istream & s, DPIOPortC<InT,OutT> &port) { 
    DPIOPortC<InT,OutT> nport(s); 
    port = nport; 
    return s; 
  }
  
  ////////////////////////////////////
  //! userlevel=Normal
  //: Plug.
  // Used for setting up inputs.
  
  template<class DataT>
  class DPPlugC {
  public:
    explicit DPPlugC(DPIPortC<DataT> &nport)
      : hold(nport.Body()),
      port(nport)
      {}
    //: Constructor.
    
    DPPlugC(const DPPlugC<DataT> &oth)
      : hold(oth.hold),
      port(oth.port)
      {}
    //: Copy constructor.
    
    inline const DPPlugC<DataT> &operator= (DPIPortC<DataT> &othport)
      { port = othport; return *this; }
    //: Assignment.
    
  private:
    DPEntityC hold; // Make sure object is not deleted.
    DPIPortC<DataT> &port;
  };
  
  /////////////////////////////
  //: Use a plug
  
  template<class DataT>
  void operator>> (DPIPortC<DataT> &source,DPPlugC<DataT> &input)  { 
    input = source; 
  }
}
#endif
