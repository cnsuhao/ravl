// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLPROB_DOMAIN_HEADER
#define RAVLPROB_DOMAIN_HEADER 1
//! rcsid="$Id$"
//! lib=RavlProb
//! author="Robert Crida"

#include "Ravl/RCHandleV.hh"
#include "Ravl/HSet.hh"
#include "Omni/Prob/RandomVariable.hh"

namespace RavlProbN {
  using namespace RavlN;

  class DomainC;

  //! userlevel=Develop
  //: Class used to represent a domain set of random variable
  class DomainBodyC
    : public RCBodyVC {
  public:
    DomainBodyC(const HSetC<RandomVariableC>& variables);
    //: Constructor
    //!param: variables - set of random variables contained in this domain

    DomainBodyC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream

    DomainBodyC(BinIStreamC &in);
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
    
    virtual ~DomainBodyC();
    //: Destructor

    bool operator==(const DomainBodyC& other) const;
    //: Check if the two domains are equal

    bool Contains(const RandomVariableC& variable) const;
    //: Check if the domain contains the specified random variable

    SizeT NumVariables() const;
    //: Get the number of variables in the domain

    const HSetC<RandomVariableC>& Variables() const;
    //: Get the random variables in the domain

    const RandomVariableC& Variable(IndexC index) const;
    //: Get a random variable by index

    IndexC Index(const RandomVariableC& variable) const;
    //: Find the index of a specified variable

    StringC ToString() const;
    //: Create a string representation of the domain

    bool operator==(const DomainC& other) const;
    //: Equality operator

    UIntT Hash() const;
    //: Hash function based set of values

  private:
    void SetVariables(const HSetC<RandomVariableC>& variables);
    //: Set the random variables in the domain

  private:
    HSetC<RandomVariableC> m_variables;
    //: The set of variables
  };

  //! userlevel=Normal
  //: Class used to represent a domain set of random variable
  //!cwiz:author
  
  class DomainC
    : public RCHandleVC<DomainBodyC>
  {
  public:
    DomainC()
    {}
    //: Default constructor makes invalid handle

    DomainC(const HSetC<RandomVariableC>& variables)
      : RCHandleVC<DomainBodyC>(new DomainBodyC(variables))
    {}

    DomainC(istream &in);
    //: Construct from stream
    //!param: in - standard input stream
    
    DomainC(BinIStreamC &in);
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
    
    bool operator==(const DomainC& other) const
    { return Body().operator==(other.Body()); }
    //: Check if the two domains are equal
    //!cwiz:author

    bool operator!=(const DomainC& other) const
    { return !Body().operator==(other.Body()); }
    //: Check if the two domains are different
    //!cwiz:author

    bool Contains(const RandomVariableC& variable) const
    { return Body().Contains(variable); }
    //: Check if the domain contains the specified random variable
    //!cwiz:author

    SizeT NumVariables() const
    { return Body().NumVariables(); }
    //: Get the number of variables in the domain

    const HSetC<RandomVariableC>& Variables() const
    { return Body().Variables(); }
    //: Get the random variables in the domain

    const RandomVariableC& Variable(IndexC index) const
    { return Body().Variable(index); }
    //: Get a random variable by index

    IndexC Index(const RandomVariableC& variable) const
    { return Body().Index(variable); }
    //: Find the index of a specified variable

    StringC ToString() const
    { return Body().ToString(); }
    //: Create a string representation of the domain
    //!cwiz:author

    UIntT Hash() const
    { return Body().Hash(); }
    //: Hash function based on name

  protected:
    DomainC(DomainBodyC &bod)
     : RCHandleVC<DomainBodyC>(bod)
    {}
    //: Body constructor. 
    
    DomainC(const DomainBodyC *bod)
     : RCHandleVC<DomainBodyC>(bod)
    {}
    //: Body constructor. 
    
    DomainBodyC& Body()
    { return static_cast<DomainBodyC &>(RCHandleVC<DomainBodyC>::Body()); }
    //: Body Access. 
    
    const DomainBodyC& Body() const
    { return static_cast<const DomainBodyC &>(RCHandleVC<DomainBodyC>::Body()); }
    //: Body Access. 
    
  };

  ostream &operator<<(ostream &s, const DomainC &obj);
  
  istream &operator>>(istream &s, DomainC &obj);

  BinOStreamC &operator<<(BinOStreamC &s, const DomainC &obj);
    
  BinIStreamC &operator>>(BinIStreamC &s, DomainC &obj);
  
}

#endif
