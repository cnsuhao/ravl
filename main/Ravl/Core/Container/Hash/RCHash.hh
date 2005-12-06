// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_RCHASH_HEADER
#define RAVL_RCHASH_HEADER 1
///////////////////////////////////////////////////
//! userlevel=Normal
//! file="Ravl/Core/Container/Hash/RCHash.hh"
//! lib=RavlCore
//! author="Charles Galambos"
//! date="06/05/1998"
//! docentry="Ravl.API.Core.Hash Tables"
//! example=WordFreq.cc
//! rcsid="$Id$"

#include "Ravl/Hash.hh"
#include "Ravl/RCWrap.hh"

namespace RavlN {
  //: Reference counted auto-resizing hash table.
  // This is a BIG OBJECT.
  // See <a href="RavlN.HashC.html">HashC</a> for more details. <p>
  // NB. This class requires the standard stream operators to
  // be implemented. ie.  operator<<(ostream &os,xxx) etc...
  
  template<class Key,class Dat >
  class RCHashC 
    : public RCWrapC<HashC<Key,Dat> > 
  {
  public:
    RCHashC(const RCHashC<Key,Dat> &oth)
      : RCWrapC<HashC<Key,Dat> >(oth) 
    {}
    //: Copy constructor.
    
    RCHashC(bool makeBod = true) 
      : RCWrapC<HashC<Key,Dat> > (makeBod,true)
    {}
    //: Default constructor.
    // Will make a body by default.
    
    RCHashC(const RCWrapC<HashC<Key,Dat> > &base)
      : RCWrapC<HashC<Key,Dat> > (base)
    {}
    //: Base constructor.
    
    RCHashC(istream &in)
      : RCWrapC<HashC<Key,Dat> > (in)
    {}
    //: Stream constructor.
    
    RCHashC(BinIStreamC &in)
      : RCWrapC<HashC<Key,Dat> > (in)
    {}
    //: Stream constructor.
    
    Dat &operator[] (const  Key &a) 
    { return this->Data()[a]; }
    //: Accesss an element.
    //!param: a - Key to lookup in table.
    //!return: Element coresponding to 'a' in the table.
    // Will create an empty element with the default constructor
    // and return a reference to it, if it doesn't already exist.
    
    const Dat &operator[] (const  Key &a) const
    { return this->Data()[a]; }
    //: Accesss an element.
    //!param: a - Key to lookup in table.
    //!return: Element coresponding to 'a' in the table.
    // The element must exist in the hash table otherwise
    // it will cause an assertion failure. Note: It will
    // just crash in optimised builds.
    
    inline bool Del(const Key &aKey)
    { return this->Data().Del(aKey); }
    //: Delete member from table.
    //!param: aKey - Key to check 
    //!return: true if Key was in hash table.
    
    inline bool IsElm(const Key &aKey) const 
    { return this->Data().IsElm(aKey); }
    //: Is key used in the table.
    //!param: aKey - Key to check 
    //!return: true if aKey is in the table.
    
    inline void Move(RCHashC<Key,Dat> &oth)
    { this->Data().Move(oth.Data()); }
    //: Move contents of another table into this one.
    // The previous contents of this table are removed.
    //!param: oth - Table to move elements from
    
    inline bool IsEmpty(void) const
    { return this->Data().IsEmpty(); }
    //: Is table empty ?
    
    inline void Empty(void)
    { this->Data().Empty(); }
    //: Empty table
    
    inline bool Insert(const Key &aKey,const Dat &data) 
    { return this->Data().Update(aKey,data); }
    //: Insert Data with Key.
    // Returns: True=Member existed already. False=New one was added.
    
    inline UIntT Size() const 
    { return this->Data().Size(); }
    //: Get number of elements in table.
    
    inline Dat *Lookup(const Key &aKey)
    { return this->Data().Lookup(aKey); }
    //: Look to see if data is present in the table.
    // Do not use, Try Lookup(key,data);
    // If data is present return a ptr to it, othersize
    // return a 0 ptr.
    
    inline const Dat *Lookup(const Key &aKey) const
    { return this->Data().Lookup(aKey); }
    //: Look to see if data is present in the table.
    // Do not use, Try Lookup(key,data);
    //!return: If data is present return a ptr to it, othersize return a 0 ptr.

    inline bool Lookup(const Key &aKey,Dat &data) const
    { return this->Data().Lookup(aKey,data); }
    //: Lookup data for key.
    //!param: aKey - Key for entry in the hash table.
    //!param: data - Place to hold results of lookup.
    //!return:true if entry is found, and is assigned to 'data', otherwise 'data' is left unmodified.
    
    inline bool Update(const Key &key,const Dat &data)
    { return this->Data().Update(key,data); }
    //: Update member of HashCable, will create new one if it doesn't exist.
    // OBSOLETE: Use Insert(key,data)
    // Returns: True=Member existed already. False=New one was added.
    
    inline Dat &Update(const Key &key)
    { return this->Data().Update(key); }
    //: Get value, add an entry created with the types default constructor if its not in the table. 
    // OBSOLETE: Use operator[].
    
    inline Dat &Access(const Key &key,const Dat &def = Dat())
    { return this->Data().Access(key,def); }
    //: Access key, if it does exists create a new bin with value 'def'
    //!param: key - Key to lookup.
    //!param: def - Default value to assign to entry if it doesn't exist.
    //!return: Reference to the new entry.
    
    void AddFrom(RCHashC<Key,Dat> &oth,bool replace = true)
    { this->Data().AddFrom(oth,replace); }
    //: Add contents of another table into this one.
    //!param: oth - Table to remove elements from.
    //!param: replace - If true replace elements from this table with those from 'oth', otherwise keep entries from this table.
    // leave other empty.  if replace is false the contents of the 
    // old table are not replace by the new entries. 
    
    bool NormaliseKey(Key &value) const
    { return this->Data().NormaliseKey(value); }
    //: Normalise an equivelent key to one used the the table.
    // This function is useful when you want to normalise the use
    // of equivlent keys (think strings.) to save memory.
    // Returns true if key exists in the table, false otherwise.
    
  };

}

#endif
