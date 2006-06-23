// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLEVALUE_HEADER
#define RAVLPROB_RANDOMVARIABLEVALUE_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/RCHandleV.hh"
#include "Ravl/Prob/RandomVariable.hh"

namespace RavlProbN {
  using namespace RavlN;

  class RandomVariableValueC;

  //! userlevel=Develop
  //: Base class for all random variable values
  class RandomVariableValueBodyC
    : public RCBodyVC {
  public:
    RandomVariableValueBodyC(const RandomVariableC& variable);
    //: Constructor
    //!param: variable - the variable that this is an instance of

    RandomVariableValueBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableValueBodyC(BinIStreamC &in);
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
    
    virtual ~RandomVariableValueBodyC();
    //: Destructor

    const RandomVariableC& RandomVariable() const;
    //: Get access to the random variable that this is an instance of

    virtual StringC ToString() const =0;
    //: Get a string representation of the values that the variable can take

    virtual bool operator==(const RandomVariableValueC& other) const=0;
    //: Equality operator

    virtual UIntT Hash() const=0;
    //: Hash function based on variable

  private:
    void SetRandomVariable(const RandomVariableC& variable);
    //: Set the random variable
    //!param: variable - the variable that this is an instance of

  private:
    RandomVariableC m_variable;
    //: The random variable
  };

  //! userlevel=Normal
  //: Base class for all random variable values
  //!cwiz:author
  
  class RandomVariableValueC
    : public RCHandleVC<RandomVariableValueBodyC>
  {
  public:
    RandomVariableValueC()
    {}
    //: Default constructor makes invalid handle

    RandomVariableValueC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableValueC(BinIStreamC &in);
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
    
    const RandomVariableC& RandomVariable() const
    { return Body().RandomVariable(); }
    //: Get access to the random variable that this is an instance of
    //!cwiz:author

    StringC ToString() const
    { return Body().ToString(); }
    //: Get a string representation of the value
    //!cwiz:author

    bool operator==(const RandomVariableValueC& other) const
    { return Body().operator==(other); }
    //: Equality operator
    //!cwiz:author

    bool operator!=(const RandomVariableValueC& other) const
    { return !Body().operator==(other); }
    //: Inequality operator
    //!cwiz:author

    UIntT Hash() const
    { return Body().Hash(); }
    //: Hash function based on variable

  protected:
    RandomVariableValueC(RandomVariableValueBodyC &bod)
     : RCHandleVC<RandomVariableValueBodyC>(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueC(const RandomVariableValueBodyC *bod)
     : RCHandleVC<RandomVariableValueBodyC>(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueBodyC& Body()
    { return static_cast<RandomVariableValueBodyC &>(RCHandleVC<RandomVariableValueBodyC>::Body()); }
    //: Body Access. 
    
    const RandomVariableValueBodyC& Body() const
    { return static_cast<const RandomVariableValueBodyC &>(RCHandleVC<RandomVariableValueBodyC>::Body()); }
    //: Body Access. 
    
  };

  ostream &operator<<(ostream &s, const RandomVariableValueC &obj);
  
  istream &operator>>(istream &s, RandomVariableValueC &obj);

  BinOStreamC &operator<<(BinOStreamC &s, const RandomVariableValueC &obj);
    
  BinIStreamC &operator>>(BinIStreamC &s, RandomVariableValueC &obj);
  
}

#endif
