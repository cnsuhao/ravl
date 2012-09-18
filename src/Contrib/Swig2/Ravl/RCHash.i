// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/Types.i"
%include typemaps.i

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/RCHash.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN
{
 template<class Key,class Dat >
  class RCHashC
  {
  public:
    RCHashC(bool makeBod = true);
    //: Default constructor.
    // Will make a body by default.

    RCHashC(SizeT nBins);

    bool Del(const Key &aKey);
    //: Delete member from table.
    //!param: aKey - Key to check 
    //!return: true if Key was in hash table.
    
    bool IsElm(const Key &aKey) const; 
    //: Is key used in the table.
    //!param: aKey - Key to check 
    //!return: true if aKey is in the table.
    
    inline void Move(RCHashC<Key,Dat> &oth);
    //: Move contents of another table into this one.
    // The previous contents of this table are removed.
    //!param: oth - Table to move elements from
    
    bool IsEmpty(void) const;
    //: Is table empty ?
    
    void Empty(void);
    //: Empty table
    
    bool Insert(const Key &aKey,const Dat &data); 
    //: Insert Data with Key.
    // Returns: True=Member existed already. False=New one was added.
    
    SizeT Size() const;
    //: Get number of elements in table.
    
    Dat *Lookup(const Key &aKey);
    //: Look to see if data is present in the table.
    // Do not use, Try Lookup(key,data);
    // If data is present return a ptr to it, otherwise
    // return a 0 ptr.
    
    const Dat *Lookup(const Key &aKey) const;
    //: Look to see if data is present in the table.
    // Do not use, Try Lookup(key,data);
    //!return: If data is present return a ptr to it, otherwise return a 0 ptr.

    bool Lookup(const Key &aKey,Dat &data) const;
    //: Lookup data for key.
    //!param: aKey - Key for entry in the hash table.
    //!param: data - Place to hold results of lookup.
    //!return:true if entry is found, and is assigned to 'data', otherwise 'data' is left unmodified.
    
    bool Update(const Key &key,const Dat &data);
    //: Update member of HashCable, will create new one if it doesn't exist.
    // OBSOLETE: Use Insert(key,data)
    // Returns: True=Member existed already. False=New one was added.
    
    Dat &Update(const Key &key);
    //: Get value, add an entry created with the types default constructor if its not in the table. 
    // OBSOLETE: Use operator[].
    
    Dat &Access(const Key &key,const Dat &dat = Dat());
    //: Access key, if it does exists create a new bin with value 'def'
    //!param: key - Key to lookup.
    //!param: def - Default value to assign to entry if it doesn't exist.
    //!return: Reference to the new entry.
    
    void AddFrom(RCHashC<Key,Dat> &oth,bool replace = true);
    //: Add contents of another table into this one.
    //!param: oth - Table to remove elements from.
    //!param: replace - If true replace elements from this table with those from 'oth', otherwise keep entries from this table.
    // leave other empty.  if replace is false the contents of the 
    // old table are not replace by the new entries. 
    
    bool NormaliseKey(Key &value) const;
    //: Normalise an equivalent key to one used the the table.
    // This function is useful when you want to normalise the use
    // of equivalent keys (think strings.) to save memory.
    // Returns true if key exists in the table, false otherwise.
    
    SizeT Hash() const;
    //: Compute hash value for table.
    
    bool operator==(const RCHashC<Key,Dat> &oth) const;
    //: Test if this hash table is equal to another.
    
    bool operator!=(const RCHashC<Key,Dat> &oth) const;
    //: Test if this hash table is different to another.

    bool operator==(const HashC<Key,Dat> &oth) const;
    //: Test if this hash table is equal to another.
    
    bool operator!=(const HashC<Key,Dat> &oth) const;
    //: Test if this hash table is different to another.

  };
}

%define RCHASH_TYPE(name, typeKey, typeVal)

%template(Hash ## name ## C) RavlN::RCHashC<typeKey, typeVal>;
%template(Hash ## name ## IterC) RavlN::RCHashIterC<typeKey, typeVal>;

%enddef

