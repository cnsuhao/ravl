// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLEVALUECONTINUOUS_HEADER
#define RAVLPROB_RANDOMVARIABLEVALUECONTINUOUS_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValue.hh"
#include "Ravl/Prob/VariableContinuous.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Implementation of a continuous random variable value
  class RandomVariableValueContinuousBodyC
    : public RandomVariableValueBodyC {
  public:
    RandomVariableValueContinuousBodyC(const VariableContinuousC& variable, RealT value);
    //: Constructor
    //!param: variable - the variable that this is an instance of
    //!param: value - the value of the random variable

    RandomVariableValueContinuousBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableValueContinuousBodyC(BinIStreamC &in);
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
    
    virtual ~RandomVariableValueContinuousBodyC();
    //: Destructor
    
    virtual StringC ToString() const;
    //: Get a string representation of the value

    virtual RealT Value() const;
    //: Get access to the value

    virtual bool operator==(const RandomVariableValueC& other) const;
    //: Equality operator

    virtual UIntT Hash() const;
    //: Hash function based on variable and value

  private:
    virtual void SetValue(RealT value);
    //: Set the value of the random variable

    VariableContinuousC VariableContinuous() const;
    //: Get access to the continuous random variable that this is an instance of

  private:
    RealT m_value;
    //: The value of the random variable
  };

  //! userlevel=Normal
  //: Implementation of a continuous random variable value
  //!cwiz:author
  
  class RandomVariableValueContinuousC
    : public RandomVariableValueC
  {
  public:
  	RandomVariableValueContinuousC()
  	{}
  	//: Default constructor makes invalid handle
  	
    RandomVariableValueContinuousC(const VariableContinuousC& variable, RealT value)
      : RandomVariableValueC(new RandomVariableValueContinuousBodyC(variable, value))
    {}
    //: Constructor
    //!param: variable - the variable that this is an instance of
    //!param: value - the value of the random variable

    RandomVariableValueContinuousC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableValueContinuousC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    RandomVariableValueContinuousC(const RandomVariableValueC& value)
      : RandomVariableValueC(dynamic_cast<const RandomVariableValueContinuousBodyC *>(BodyPtr(value)))
    {}
    //: Upcast constructor
    // Creates an invalid handle if types don't match
    
    RealT Value() const
    { return Body().Value(); }
    //: Get access to the value

  protected:
    RandomVariableValueContinuousC(RandomVariableValueContinuousBodyC &bod)
     : RandomVariableValueC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueContinuousC(const RandomVariableValueContinuousBodyC *bod)
     : RandomVariableValueC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueContinuousBodyC& Body()
    { return static_cast<RandomVariableValueContinuousBodyC &>(RandomVariableValueC::Body()); }
    //: Body Access. 
    
    const RandomVariableValueContinuousBodyC& Body() const
    { return static_cast<const RandomVariableValueContinuousBodyC &>(RandomVariableValueC::Body()); }
    //: Body Access. 
    
  };

}

#endif
