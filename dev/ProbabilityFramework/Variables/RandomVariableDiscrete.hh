// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_RANDOMVARIABLEDISCRETE_HEADER
#define RAVLPROB_RANDOMVARIABLEDISCRETE_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/Prob/Variable.hh"
#include "Ravl/HSet.hh"

namespace RavlProbN {
  using namespace RavlN;

  //! userlevel=Develop
  //: Implementation of a discrete random variable
  class RandomVariableDiscreteBodyC
    : public VariableBodyC {
  public:
    RandomVariableDiscreteBodyC(const StringC& name, const HSetC<StringC>& values);
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face
    //!param: values - countable set of mutually exclusive values that the variable can take

    RandomVariableDiscreteBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    RandomVariableDiscreteBodyC(BinIStreamC &in);
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
    
    virtual ~RandomVariableDiscreteBodyC();
    //: Destructor
    
    virtual StringC ToString() const;
    //: Get a string representation of the values that the variable can take

    SizeT NumValues() const;
    //: Get access to the number of legal values

    const HSetC<StringC>& Values() const;
    //: Get access to the set of legal values

    const StringC& Value(IndexC index) const;
    //: Get the value for a particular index

    IndexC Index(const StringC& value) const;
    //: Lookup an index for a value

  protected:
    RandomVariableDiscreteBodyC(const StringC& name);
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face
    //!param: values - countable set of mutually exclusive values that the variable can take

    void SetValues(const HSetC<StringC>& values);
    //: Set the countable set of mutually exclusive value that this variable can take

  private:
    SizeT m_numValues;
    //: Cached count of the number of values in the set

    HSetC<StringC> m_values;
    //: Countable set of mutually exclusive values that the variable can take
  };

  //! userlevel=Normal
  //: Implementation of a discrete random variable
  //!cwiz:author
  
  class RandomVariableDiscreteC
    : public VariableC
  {
  public:
    RandomVariableDiscreteC()
    {}
    //: Default constructor makes invalid handle

    RandomVariableDiscreteC(const StringC& name, const HSetC<StringC>& values)
      : VariableC(new RandomVariableDiscreteBodyC(name, values))
    {}
    //: Constructor
    //!param: name - convention is that it starts with a Capital letter, eg Face
    //!param: values - countable set of mutually exclusive values that the variable can take

    RandomVariableDiscreteC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    RandomVariableDiscreteC(BinIStreamC &in);
    //: Construct from binary stream
    //!param: in - binary input stream
    
    RandomVariableDiscreteC(const VariableC& variable)
      : VariableC(dynamic_cast<const RandomVariableDiscreteBodyC *>(BodyPtr(variable)))
    {}
    //: Upcast constructor
    // Creates an invalid handle if types don't match
    
    SizeT NumValues() const
    { return Body().NumValues(); }
    //: Get access to the number of legal values

    const HSetC<StringC>& Values() const
    { return Body().Values(); }
    //: Get access to the set of legal values

    const StringC& Value(IndexC index) const
    { return Body().Value(index); }
    //: Get the value for a particular index

    IndexC Index(const StringC& value) const
    { return Body().Index(value); }
    //: Lookup an index for a value

  protected:
    RandomVariableDiscreteC(RandomVariableDiscreteBodyC &bod)
     : VariableC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableDiscreteC(const RandomVariableDiscreteBodyC *bod)
     : VariableC(bod)
    {}
    //: Body constructor. 
    
    RandomVariableDiscreteBodyC& Body()
    { return static_cast<RandomVariableDiscreteBodyC &>(VariableC::Body()); }
    //: Body Access. 
    
    const RandomVariableDiscreteBodyC& Body() const
    { return static_cast<const RandomVariableDiscreteBodyC &>(VariableC::Body()); }
    //: Body Access. 
    
  };

}

#endif
