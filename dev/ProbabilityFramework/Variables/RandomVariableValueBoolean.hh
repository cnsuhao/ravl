// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLEVALUEBOOLEAN_HEADER
#define RAVLPROB_RANDOMVARIABLEVALUEBOOLEAN_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/RandomVariableValueDiscrete.hh"
#include "Ravl/Prob/RandomVariableBoolean.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Implementation of a boolean random variable value
  class RandomVariableValueBooleanBodyC
    : public RandomVariableValueDiscreteBodyC {
  public:
    RandomVariableValueBooleanBodyC(const RandomVariableBooleanC& variable, bool value);
    //: Constructor
    //!param: variable - the variable that this is an instance of
    //!param: value - the value of the random variable

    RandomVariableValueBooleanBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableValueBooleanBodyC(BinIStreamC &in);
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
    
    virtual ~RandomVariableValueBooleanBodyC();
    //: Destructor
    
    bool BooleanValue() const;
    //: Get access to the value

  private:
    void SetBooleanValue(bool value);
    //: Set the value of the random variable

    virtual void SetValue(const StringC& value);
    //: Set the value of the random variable

    RandomVariableBooleanC RandomVariableBoolean() const;
    //: Get access to the boolean random variable that this is an instance of

  private:
    bool m_booleanValue;
    //: The value of the random variable
  };

  //! userlevel=Normal
  //: Implementation of a boolean random variable value
  //!cwiz:author
  
  class RandomVariableValueBooleanC
    : public RandomVariableValueDiscreteC
  {
  public:
    RandomVariableValueBooleanC(const RandomVariableBooleanC& variable, bool value)
      : RandomVariableValueDiscreteC(new RandomVariableValueBooleanBodyC(variable, value))
    {}
    //: Constructor
    //!param: variable - the variable that this is an instance of
    //!param: value - the value of the random variable

    RandomVariableValueBooleanC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableValueBooleanC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    bool BooleanValue() const
    { return Body().BooleanValue(); }
    //: Get access to the value

  protected:
    RandomVariableValueBooleanC(RandomVariableValueBooleanBodyC &bod)
     : RandomVariableValueDiscreteC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueBooleanC(const RandomVariableValueBooleanBodyC *bod)
     : RandomVariableValueDiscreteC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableValueBooleanBodyC& Body()
    { return static_cast<RandomVariableValueBooleanBodyC &>(RandomVariableValueDiscreteC::Body()); }
    //: Body Access. 
    
    const RandomVariableValueBooleanBodyC& Body() const
    { return static_cast<const RandomVariableValueBooleanBodyC &>(RandomVariableValueDiscreteC::Body()); }
    //: Body Access. 
    
  };

}

#endif
