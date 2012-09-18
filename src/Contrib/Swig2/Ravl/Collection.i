// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/Macros.i"
%include "Ravl/Swig2/Point2d.i"
%include "Ravl/Swig2/Vector.i"
%include "Ravl/Swig2/DList.i"
%include "Ravl/Swig2/RCHash.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif


#include "Ravl/Collection.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/Vector.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  template<class DataT>
  class CollectionC 
  {
  public:
    
    %extend {
    	CollectionC(UIntT size=10) {
    		return new RavlN::CollectionC<DataT>(size);
    	}
    }
    
    CollectionC(SizeT maxSize,SizeT used = 0);
    //: Create an empty collection.
    // maxSize should be set to maximum number of elements the collection 
    // will contain.  'used' is the number of elements to be in the collection 
    // at the time of creation.
    
    CollectionC(const SArray1dC<DataT> &dat);
    //: Create a collection from an array of data.
    
    CollectionC(const DListC<DataT> &list);
    //: Construct collection from a list.
    
    CollectionC<DataT> Copy() const;
    //: Create a copy of this collection.
    
    UIntT Insert(const DataT &dat);
    //: Add a data item to the collection.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection. <br>
    // The index at which the item was placed is returned.
    
    //UIntT Insert(const Array1dC<DataT> &dat);
    //: Add an array of data items to the collection.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection. <p>
    // The first index at which the items were placed is returned.
    
    UIntT Append(const DataT &dat);
    //: Add a data item to the end of the collection.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection. <p>
    // The index at which the item was placed is returned.
    
    //UIntT Append(const Array1dC<DataT> &dat);
    //: Add an array of data items to the end of the collection.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection. <p>
    // The first index at which the items were placed is returned.
    
    UIntT InsertRandom(const DataT &dat);
    //: Add a data item to the collection in a random place.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection.
    
    void Delete(IndexC ind);
    //: Remove item at 'ind' from the collection.
    
    void operator+=(const DataT &dat);
    //: Add data item to the collection.
    //  NB. This may cause the storage array to 
    // be reallocated which will invalidate any iterators
    // held on the collection.
    
    DataT Pick();
    //: Pick a random item from the collection.
    // the element will be removed from the set.
    // It is the users responsibility to ensure the
    // set is not empty when this method is called.
    // See 'IsEmpty()'
    
    CollectionC<DataT> Shuffle() const;
    //: Create a shuffled version of this collection.
    
    void ShuffleIP();
    //: Shuffle collection in place.
    
    void Merge(const CollectionC<DataT> &x);
    //: Merge collection 'x' into this one.
    
    void Merge(const SArray1dC<DataT> &x);
    //: Merge of array 'x' into this collection.
    
    bool IsEmpty() const;
    //: Test if collection is empty.
    
    void Empty();
    //: Empty the collection of all its contents.
    
    SizeT Size() const;
    //: Returns the number of data elements in the collection.
    
    SArray1dC<DataT> SArray1d();
    //: Access data as array.
    // Note: The returned array is a direct access
    // to the internal data structure, no operations
    // that modify the collection should be performed
    // while its in use.

    const SArray1dC<DataT> SArray1d() const;
    //: Access data as array.
    // Note: The returned array is a direct access
    // to the internal data structure, no operations
    // that modify the collection should be performed
    // while its in use.

    CollectionC<DataT> Split(SizeT ne);
    //: Split the collection in two
    // return a random set of 'ne' elements from this one.
    // ne must be smaller than the size of the collection.
    
    CollectionC<DataT> Sample(SizeT ne) const;
    //: Take a random sample from the collection.
    // This collection is not modified.  There is no
    // garantee that an element will be picked only once.
    // 'ne' may be bigger than the size of this collection.
    
    //DataT &KthHighest(UIntT k);
    //: k'th highest element of a collection
    // k should be between 0 and Size()-1. For the median use k = Size()/2.
    // The DataT type class must have the < and > operators defined on them.
    // This function will reorder the collection so that elements 0...k-1
    // are <= element k and elements k+1...Size()-1 are >= element k. 
    // Based on algorithm in Numerical Recipes in C, second edition.
    // <p>
    // Note: This method re-orders the contents of the collection.


    DataT &Last();
    //: Access last element in the collection.

    const DataT &Last() const;
    //: Access last element in the collection.
    
    DataT &First();
    //: Access first element in the collection.
    
    const DataT &First() const;
    //: Access first element in the collection.
    
    %extend 
	{
	  inline const DataT & __getitem__(size_t i) const { return (*self)[i]; }
	  
	  inline void __setitem__(size_t i, const  DataT & v) { (*self)[i] = v; }
	  
	  __STR__();

    }
    
  };
 
}

%template(CollectionUInt) RavlN::CollectionC<RavlN::UIntT>;
%template(CollectionReal) RavlN::CollectionC<RavlN::RealT>;
%template(CollectionFloat) RavlN::CollectionC<RavlN::FloatT>;
%template(CollectionIndexC) RavlN::CollectionC<RavlN::IndexC>;
%template(CollectionPoint2dC) RavlN::CollectionC<RavlN::Point2dC>;
%template(CollectionVectorC) RavlN::CollectionC<RavlN::VectorC>;
%template(RCHashStringPointsC) RavlN::RCHashC<RavlN::StringC, RavlN::CollectionC<RavlN::Point2dC> >;
