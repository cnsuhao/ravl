// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_PROPOSITION_HEADER
#define RAVLPROB_PROPOSITION_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/RCHandleV.hh"
#include "Ravl/Prob/Domain.hh"
#include "Ravl/Prob/RandomVariableValue.hh"
#include "Ravl/HSet.hh"

namespace RavlProbN {
  using namespace RavlN;

  class PropositionC;

  //! userlevel=Develop
  //: Class used to represent a proposition in a domain of random variables
  class PropositionBodyC
    : public RCBodyVC {
  public:
    PropositionBodyC(const DomainC& domain, const HSetC<RandomVariableValueC>& values);
    //: Constructor
    //!param: domain - the domain for the proposition
    //!param: values - list of random variables values contained in this proposition

    PropositionBodyC(const PropositionBodyC& other, const RandomVariableValueC& value);
    //: Extended constructor
    //!param: other = another proposition
    //!param: value - a value to extend the proposition with

    PropositionBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    PropositionBodyC(BinIStreamC &in);
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
    
    virtual ~PropositionBodyC();
    //: Destructor

    StringC ToString() const;
    //: Create a string representation of the domain

    StringC LotteryName() const;
    //: Create a lottery name for this proposition

    const DomainC& Domain() const;
    //: Get the domain

    SizeT NumValues() const;
    //: Get the number of values in the proposition

    const HSetC<RandomVariableValueC>& Values() const;
    //: Get the random variable values in the domain

    const RandomVariableValueC& Value(IndexC index) const;
    //: Get a random variable value by index

    PropositionC SubProposition(const DomainC& subDomain) const;
    //: Create a proposition for a subdomain

    bool operator==(const PropositionBodyC& other) const;
    //: Equality operator

    UIntT Hash() const;
    //: Hash function based set of values

  private:
    void SetDomain(const DomainC& domain);
    //: Set the domain

    void SetValues(const HSetC<RandomVariableValueC>& values);
    //: Set the random variable values in the domain

  private:
    DomainC m_domain;
    //: The domain of the proposition

    HSetC<RandomVariableValueC> m_values;
    //: The set of variables
  };

  //! userlevel=Normal
  //: Class used to represent a proposition in a domain of random variables
  //!cwiz:author
  
  class PropositionC
    : public RCHandleVC<PropositionBodyC>
  {
  public:
    PropositionC()
    {}
    //: Default constructor makes invalid handle

    PropositionC(const DomainC& domain, const HSetC<RandomVariableValueC>& values)
      : RCHandleVC<PropositionBodyC>(new PropositionBodyC(domain, values))
    {}

    PropositionC(const PropositionC& other, const RandomVariableValueC& value)
      : RCHandleVC<PropositionBodyC>(new PropositionBodyC(other.Body(), value))
    {}
    //: Extended constructor
    //!param: other = another proposition
    //!param: value - a value to extend the proposition with

    PropositionC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    PropositionC(BinIStreamC &in);
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
    
    StringC ToString() const
    { return Body().ToString(); }
    //: Create a string representation of the domain
    //!cwiz:author

    StringC LotteryName() const
    { return Body().LotteryName(); }
    //: Create a lottery name for this proposition
    //!cwiz:author

    const DomainC& Domain() const
    { return Body().Domain(); }
    //: Get the domain

    SizeT NumValues() const
    { return Body().NumValues(); }
    //: Get the number of values in the proposition

    const HSetC<RandomVariableValueC>& Values() const
    { return Body().Values(); }
    //: Get the random variable values in the domain
    //!cwiz:author

    const RandomVariableValueC& Value(IndexC index) const
    { return Body().Value(index); }
    //: Get a random variable value by index
    //!cwiz:author

    PropositionC SubProposition(const DomainC& subDomain) const
    { return Body().SubProposition(subDomain); }
    //: Create a proposition for a subdomain
    //!cwiz:author

    bool operator==(const PropositionC& other) const
    { return Body().operator==(other.Body()); }
    //: Equality operator
    //!cwiz:author

    bool operator!=(const PropositionC& other) const
    { return !Body().operator==(other.Body()); }
    //: Inequality operator
    //!cwiz:author

    UIntT Hash() const
    { return Body().Hash(); }
    //: Hash function based on name

  protected:
    PropositionC(PropositionBodyC &bod)
     : RCHandleVC<PropositionBodyC>(bod)
    {}
    //: Body constructor. 
    
    PropositionC(const PropositionBodyC *bod)
     : RCHandleVC<PropositionBodyC>(bod)
    {}
    //: Body constructor. 
    
    PropositionBodyC& Body()
    { return static_cast<PropositionBodyC &>(RCHandleVC<PropositionBodyC>::Body()); }
    //: Body Access. 
    
    const PropositionBodyC& Body() const
    { return static_cast<const PropositionBodyC &>(RCHandleVC<PropositionBodyC>::Body()); }
    //: Body Access. 
    
  };

  ostream &operator<<(ostream &s, const PropositionC &obj);
  
  istream &operator>>(istream &s, PropositionC &obj);

  BinOStreamC &operator<<(BinOStreamC &s, const PropositionC &obj);
    
  BinIStreamC &operator>>(BinIStreamC &s, PropositionC &obj);
  
}

#endif
