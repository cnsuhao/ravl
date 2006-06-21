// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLEBOOLEAN_HEADER
#define RAVLPROB_RANDOMVARIABLEBOOLEAN_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Omni/Prob/RandomVariableDiscrete.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Implementation of a boolean random variable
  class RandomVariableBooleanBodyC
    : public RandomVariableDiscreteBodyC {
  public:
    RandomVariableBooleanBodyC(const StringC& name);
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face

    RandomVariableBooleanBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableBooleanBodyC(BinIStreamC &in);
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
    
    virtual ~RandomVariableBooleanBodyC();
    //: Destructor

    const StringC& Value(bool value) const;
    //: Get the name used for value

  private:
    void SetValueNames();
    //: Sets trueValue and falseValue based on the variable name

  private:
    StringC m_trueValue;
    //: The name used for true values

    StringC m_falseValue;
    //: The name used for false values
  };

  //! userlevel=Normal
  //: Implementation of a boolean random variable
  //!cwiz:author
  
  class RandomVariableBooleanC
    : public RandomVariableDiscreteC
  {
  public:
    RandomVariableBooleanC(const StringC& name)
      : RandomVariableDiscreteC(new RandomVariableBooleanBodyC(name))
    {}
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face

    RandomVariableBooleanC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableBooleanC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    RandomVariableBooleanC(const RandomVariableC& variable)
      : RandomVariableDiscreteC(dynamic_cast<const RandomVariableBooleanBodyC *>(BodyPtr(variable)))
    {}
    //: Upcast constructor
    // Creates an invalid handle if types don't match
    
    const StringC& Value(bool value) const
    { return Body().Value(value); }
    //: Get the name used for bool value

  protected:
    RandomVariableBooleanC(RandomVariableBooleanBodyC &bod)
     : RandomVariableDiscreteC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableBooleanC(const RandomVariableBooleanBodyC *bod)
     : RandomVariableDiscreteC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableBooleanBodyC& Body()
    { return static_cast<RandomVariableBooleanBodyC &>(RandomVariableDiscreteC::Body()); }
    //: Body Access. 
    
    const RandomVariableBooleanBodyC& Body() const
    { return static_cast<const RandomVariableBooleanBodyC &>(RandomVariableDiscreteC::Body()); }
    //: Body Access. 
    
  };

}

#endif
