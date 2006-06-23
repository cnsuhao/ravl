// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLE_HEADER
#define RAVLPROB_RANDOMVARIABLE_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/RCHandleV.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/String.hh"

namespace RavlProbN {
  using namespace RavlN;

  class RandomVariableC;

  //! userlevel=Develop
  //: Base class for all random variable types
  class RandomVariableBodyC
    : public RCBodyVC {
  public:
    RandomVariableBodyC(const StringC& name);
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face

    RandomVariableBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableBodyC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    virtual bool Save (ostream &out) const;
    //: Writes object to stream, can be loaded using stream constructor
    //!param: out - standard output stream
    //!return: true if the object was successfully saved

    virtual bool Save (BinOStreamC &out) const;
    //: Writes object to stream, can be loaded using binary stream constructor
    //!param: out - binary output stream
    //!return: true if the object was successfully saved
    
    virtual ~RandomVariableBodyC();
    //: Destructor

    const StringC& Name() const;
    //: Get access to the name of the random variable

    virtual StringC ToString() const =0;
    //: Get a string representation of the values that the variable can take
    
    bool operator==(const RandomVariableC& other) const;
    //: Equality operator

    UIntT Hash() const;
    //: Hash function based on name

  private:
    void SetName(const StringC& name);
    //: Set the name of the random variable
    //!param: name - convention is that it starts with a Capital letter, eg Face

  private:
    StringC m_name;
    //: The name of the random variable
  };

  //! userlevel=Normal
  //: Base class for all random variable types
  //!cwiz:author
  
  class RandomVariableC
    : public RCHandleVC<RandomVariableBodyC>
  {
  public:
    RandomVariableC()
    {}
    //: Default constructor makes invalid handle

    RandomVariableC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    bool Save(ostream &out) const
    { return Body().Save(out); }
    //: Writes object to stream, can be loaded using stream constructor 
    //!param: out - standard output stream
    //!return: true if the object was successfully saved
    //!cwiz:author
    
    bool Save(BinOStreamC &out) const
    { return Body().Save(out); }
    //: Writes object to stream, can be loaded using binary stream constructor 
    //!param: out - binary output stream
    //!return: true if the object was successfully saved
    //!cwiz:author
    
    const StringC& Name() const
    { return Body().Name(); }
    //: Get access to the name of the random variable
    //!cwiz:author

    StringC ToString()
    { return Body().ToString(); }
    //: Get a string representation of the values that the variable can take
    //!cwiz:author

    bool operator==(const RandomVariableC& other) const
    { return Body().operator==(other); }
    //: Equality operator
    //!cwiz:author

    bool operator!=(const RandomVariableC& other) const
    { return !Body().operator==(other); }
    //: Inequality operator
    //!cwiz:author

    UIntT Hash() const
    { return Body().Hash(); }
    //: Hash function based on name

  protected:
    RandomVariableC(RandomVariableBodyC &bod)
     : RCHandleVC<RandomVariableBodyC>(bod)
    {}
    //: Body constructor. 
    
    RandomVariableC(const RandomVariableBodyC *bod)
     : RCHandleVC<RandomVariableBodyC>(bod)
    {}
    //: Body constructor. 
    
    RandomVariableBodyC& Body()
    { return static_cast<RandomVariableBodyC &>(RCHandleVC<RandomVariableBodyC>::Body()); }
    //: Body Access. 
    
    const RandomVariableBodyC& Body() const
    { return static_cast<const RandomVariableBodyC &>(RCHandleVC<RandomVariableBodyC>::Body()); }
    //: Body Access. 
    
  };

  ostream &operator<<(ostream &s, const RandomVariableC &obj);
  
  istream &operator>>(istream &s, RandomVariableC &obj);

  BinOStreamC &operator<<(BinOStreamC &s, const RandomVariableC &obj);
    
  BinIStreamC &operator>>(BinIStreamC &s, RandomVariableC &obj);
  
}

#endif
