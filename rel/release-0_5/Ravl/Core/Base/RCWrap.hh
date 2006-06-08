// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_WRAP_HEADER
#define RAVL_WRAP_HEADER 1
//////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! docentry="Ravl.Core.Reference Counting"
//! file="Ravl/Core/Base/RCWrap.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Charles Galambos"
//! date="06/05/1998"

#include "Ravl/RefCounter.hh"
#include "Ravl/RCAbstract.hh"
#if RAVL_HAVE_RTTI
#if RAVL_HAVE_ANSICPPHEADERS
#include <typeinfo>
#else
#include <typeinfo.h>
#endif
#endif

//: Ravl library namespace.

namespace RavlN {  
  
  //! userlevel=Develop
  //: RCWrapped object base class.
  
  class RCWrapBaseBodyC 
    : public RCBodyVC
  {
  public:
    RCWrapBaseBodyC()
      {}
    //: Default constructor.

#if RAVL_HAVE_RTTI    
    virtual const type_info &DataType() const;
    //: Get type of wrapped object.
#endif
  };
  
  //! userlevel=Advanced
  //: Abstract wrapped object handle.
  
  class RCWrapAbstractC
    : public RCHandleC<RCWrapBaseBodyC>
  {
  public:
    RCWrapAbstractC()
      {}
    //: Default constructor.
    // Creates an invalid handle.

    RCWrapAbstractC(const RCAbstractC &val)
      : RCHandleC<RCWrapBaseBodyC>(val)
      {}
    //:  Constructor from an abstract.
    
  protected:
    RCWrapAbstractC(RCWrapBaseBodyC &bod)
      : RCHandleC<RCWrapBaseBodyC>(bod)
      {}
    //: Body constructor.

    RCWrapBaseBodyC &Body()
      { return RCHandleC<RCWrapBaseBodyC>::Body(); }
    //: Access body of object.

    const RCWrapBaseBodyC &Body() const
      { return RCHandleC<RCWrapBaseBodyC>::Body(); }
    //: Access body of object.
    
  public:    
    
#if RAVL_HAVE_RTTI
    const type_info &DataType() const
      { return Body().DataType(); }
    //: Get type of wrapped object.
#endif
  };
  
  //! userlevel=Develop
  //: RCWrapper body.
  
  template<class DataT>
  class RCWrapBodyC 
    : public RCWrapBaseBodyC
  {
  public:
    RCWrapBodyC()
      {}
    //: Default constructor.

    RCWrapBodyC(const DataT &val)
      : data(val)
      {}
    //: Constructor.
    
    RCWrapBodyC(istream &in) {
      in >> data;
    }
    //: Construct from a stream.
    
    DataT &Data()
      { return data; }
    //: Access data.

    const DataT &Data() const
      { return data; }
    //: Access data.
    
#if RAVL_HAVE_RTTI   
    virtual const type_info &DataType() const
      { return typeid(DataT); }
    //: Get type of wrapped object.
#endif
    
  protected:
    DataT data;
  };
  
  //! userlevel=Advanced
  //: RCWrapper handle.
  
  template<class DataT>
  class RCWrapC
    : public RCWrapAbstractC
  {
  public:
    RCWrapC()
      {}
    //: Default constructor.
    // Creates an invalid handle.

    RCWrapC(bool makebod,bool){
      if(makebod)
	*this = RCWrapC(DataT());
    }
    //: Default constructor.
    // Creates an invalid handle.
    
    RCWrapC(const DataT &dat)
      : RCWrapAbstractC(*new RCWrapBodyC<DataT>(dat))
      {}
    //: Construct from an instance.
    // Uses the copy constructor to creat a reference
    // counted copy of 'dat.
    
    RCWrapC(const RCWrapAbstractC &val,bool v)
      : RCWrapAbstractC(val)
      {
	if(dynamic_cast<RCWrapBodyC<DataT> *>(&RCWrapAbstractC::Body()) == 0)
	  Invalidate();
      }
    //: Construct from an abstract handle.
    // if the object types do not match, an invalid handle
    // is created.
    
    RCWrapC(const RCAbstractC &val)
      : RCWrapAbstractC(val) 
      {
	if(dynamic_cast<RCWrapBodyC<DataT> *>(&RCWrapAbstractC::Body()) == 0)
	  Invalidate();	
      }
    //: Construct from an abstract handle.
    
    RCWrapC(istream &in)
      : RCWrapAbstractC(*new RCWrapBodyC<DataT>(in))
      {}
    //: Construct from a stream.
    
  protected:
    RCWrapC(RCWrapBodyC<DataT> &bod)
      : RCWrapAbstractC(bod)
      {}
    //: Body constructor.
    
    RCWrapBodyC<DataT> &Body()
      { return static_cast<RCWrapBodyC<DataT> &>(RCWrapAbstractC::Body()); }
    //: Body access.
    
    const RCWrapBodyC<DataT> &Body() const
      { return static_cast<const RCWrapBodyC<DataT> &>(RCWrapAbstractC::Body()); }
    //: Constant body access.
    
  public:
    RCWrapC<DataT> Copy() const 
      { return RCWrapC<DataT>(Body().Data()); }
    //: Make a copy of this handle.
    // NB. This assumes the wrapped object is SMALL, and so
    // just using the copy constructor is sufficent.
    
    DataT &Data()
      { return Body().Data(); }
    //: Access data.

    const DataT &Data() const
      { return Body().Data(); }
    //: Access data.

    operator DataT &() 
      { return Body().Data(); }
    //: Default conversion to data type.

    operator const DataT &() const 
      { return Body().Data(); }
    //: Default conversion to data type.
  };
  
  template<class DataT>
  ostream &operator<<(ostream &strm,const RCWrapC<DataT> &data) {
    RavlAssert(data.IsValid());
    strm << data.Data();
    return strm;
  }
  //: ostream operator.

  template<class DataT>
  istream &operator>>(istream &strm,RCWrapC<DataT> &data) {
    DataT tmp;
    strm >> tmp;
    data = RCWrapC<DataT>(tmp);
    return strm;
  }
  //: istream operator.
  
}



#endif