// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_DLIST_HEADER
#define RAVL_DLIST_HEADER 1
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! docentry="Ravl.Core.Lists"
//! file="Ravl/Core/Container/DList/DList.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Charles Galambos"
//! example=exDList.cc

// DListC is based on code written by Radek Marik.

#include "Ravl/DLink.hh"
#include "Ravl/Assert.hh"
#include "Ravl/RefCounter.hh"

//: Ravl global namespace.

namespace RavlN {
  
  template<class DataT> class DLIterC;
  template<class DataT> class DListC;
  template<class DataT> class DListBodyC;
  template<class DataT> ostream &operator<<(ostream &,const DListBodyC<DataT> &);
  template<class DataT> istream &operator>>(istream &,DListBodyC<DataT> &);
  //! userlevel=Develop
  //: Double linked list body.
  
  template<class DataT>
  class DListBodyC
    : public DLinkHeadC,
      public RCBodyVC
  {
  public:
    DListBodyC()
      {}
    //: Default constructor.

    ~DListBodyC()
      { Empty(); }
    //: Destructor.
    
    RCBodyVC &Copy() const;
    //: Make copy of body.
    // This should be provided in derived classes.
    // this funtion will issue an assertion failure if called.
    
    void Empty() {
      while(&head.Next() != &head)
	DListBodyC<DataT>::Delete(head.Next());
    }
    //: Empty the list of all contents
    
    void InsFirst(const DataT &dat)
      { DLinkHeadC::InsFirst(*new DLinkDataC<DataT>(dat)); }
    //: Insert element into the begining of the list.

    void InsLast(const DataT &dat)
      { DLinkHeadC::InsLast(*new DLinkDataC<DataT>(dat)); }
    //: Insert element into the end of the list.

    DataT PopFirst()  { 
      RavlAssert(!IsEmpty());
      DataT tmp(FirstLink().Data());
      DListBodyC<DataT>::Delete(head.Next());  
      return tmp;
    }
    //: Pop item off beginning of this list.

    DataT PopLast()  { 
      RavlAssert(!IsEmpty());
      DataT tmp(LastLink().Data());
      DListBodyC<DataT>::Delete(head.Prev());
      return tmp;
    }
    //: Pop item off end of list.
    
    void operator+=(const DataT &dat)
      { InsLast(dat); }
    //: Add 'dat' to end of this list.
    
    void operator+=(const DListC<DataT> &dat);
    //: Copy contents of other list to end of this one.
    
    void MoveFirst(DListC<DataT> & lst)
      { DLinkHeadC::MoveFirst(lst.Body()); }
    //: Move the entire contents of 'lst' to the front of this one.
    // this leaves 'lst' empty.
    
    void MoveLast(DListC<DataT> & lst)
      { DLinkHeadC::MoveLast(lst.Body()); }
    //: Move the entire contents of 'lst' to the end of this one.
    // this leaves 'lst' empty.
    
    void MoveFirst(DLIterC<DataT> & at) { 
      RavlAssert(at);
      DLinkC &nxt = at.place->Prev();
      DLinkHeadC::MoveFront(*at.place); 
      at.place = &nxt;
    }
    //: Move the single item 'at' to the front of this list.
    // Leaves iterator pointing to the previous element.
    
    void MoveLast(DLIterC<DataT> & at) { 
      RavlAssert(at);
      DLinkC &nxt = at.place->Prev(); 
      DLinkHeadC::MoveLast(*at.place); 
      at.place = &nxt;
    }
    //: Move the single item 'at' to the end of this list.
    // Leaves iterator pointing to the previous element.
    
    typedef bool (*MergeSortCmpT)(DataT &l1,DataT &l2);
    //: Comparison function for merge sort.
    
    bool operator==(const DListBodyC<DataT> &oth) const;
    //: Test if lists are identical.
    // assumes '==' is defined for 'DataT'
    
    bool operator!=(const DListBodyC<DataT> &oth) const
      { return !((*this) == oth); }
    //: Test if lists are different.
    // assumes '==' is defined for 'DataT'
    
  protected:    
    static bool MergeSortHelpCmp(DLinkC *l1,DLinkC *l2,void *dat) {
      return ((MergeSortCmpT) dat)(static_cast<DLinkDataC<DataT> &>(*l1).Data(),
				   static_cast<DLinkDataC<DataT> &>(*l2).Data());
    }
    //: Comparison helper function.
    
  public:
    //MergeSortCmpT
    void MergeSort(bool (*cmp)(DataT &l1,DataT &l2)) // 
      { DLinkHeadC::MergeSort(&DListBodyC<DataT>::MergeSortHelpCmp,(void *) cmp); }
    //: Merge sort the list with comparison function 'cmp'.
    
    DataT &First() { 
      RavlAssert(!IsEmpty());
      return FirstLink().Data(); 
    }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    DataT &Last() { 
      RavlAssert(!IsEmpty());
      return LastLink().Data(); 
    }
    //: Get last element in list.
    // NB. List MUST be non-empty.
    
    const DataT &First() const { 
      RavlAssert(!IsEmpty());
      return FirstLink().Data(); 
    }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    const DataT &Last() const { 
      RavlAssert(!IsEmpty());
      return LastLink().Data(); 
    }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
  protected:
    static void Delete(DLinkC &elm) 
      { delete &static_cast<DLinkDataC<DataT> &>(elm); }
    //: Unlink and delete an element from the list.
    // The delete will unkink the node automaticly.
    
    DLinkDataC<DataT> &FirstLink()
      { return static_cast<DLinkDataC<DataT> &>(head.Next()); }
    //: Get first link in list.
    
    DLinkDataC<DataT> &LastLink()
      { return static_cast<DLinkDataC<DataT> &>(head.Prev()); }
    //: Get the last ilink in the list.
    
    const DLinkDataC<DataT> &FirstLink() const
      { return static_cast<DLinkDataC<DataT> &>(head.Next()); }
    //: Get first link in list.
    
    const DLinkDataC<DataT> &LastLink() const
      { return static_cast<const DLinkDataC<DataT> &>(head.Prev()); }
    //: Get the last ilink in the list.
    
    DLinkC &Head()
      { return DLinkHeadC::Head(); }
    //: Get head of list.

    const DLinkC &Head() const
      { return DLinkHeadC::Head(); }
    //: Get head of list.
    
    friend class DListC<DataT>;
    friend class DLIterC<DataT>;
    
#if RAVL_NEW_ANSI_CXX_DRAFT
    friend ostream &operator<< <DataT>(ostream &strm,const DListBodyC<DataT> &lst); 
#else
    friend ostream &operator<< (ostream &strm,const DListBodyC<DataT> &lst); 
#endif
  };

  template<class DataT>
  ostream &operator<<(ostream &out,const DListBodyC<DataT> &lst);
  //: Send to stream.
  
  template<class DataT>
  istream &operator>>(istream &out,DListBodyC<DataT> &lst);
  //: Read from stream.
  
  template<class DataT>
  ostream &operator<<(ostream &strm,const DListC<DataT> &lst);
  //: Send to stream.
  
  template<class DataT>
  istream &operator>>(istream &strm,DListC<DataT> &lst);
  //: Read from stream.
  
  //! userlevel=Normal
  //: Double linked List 
  // This is a refrence counted, doubly linked list.
  
  template<class DataT>
  class DListC 
    : public RCHandleC<DListBodyC<DataT> > 
  {
  public:
    typedef DLIterC<DataT> iterator;
    //: Stl compatability
    
    DListC()
      : RCHandleC<DListBodyC<DataT> >(*new DListBodyC<DataT>())
      {}
    //: Default constructor.
    // This creates an empty list.
    
  protected:
    DListC(DListBodyC<DataT> &bod)
      : RCHandleC<DListBodyC<DataT> >(bod)
      {}
    //: Body constructor.
    
    DListBodyC<DataT> &Body()
      { return RCHandleC<DListBodyC<DataT> >::Body(); }
    //: Access body.

    const DListBodyC<DataT> &Body() const
      { return RCHandleC<DListBodyC<DataT> >::Body(); }
    //: Constant access to body.
    
    DLinkC &Head()
      { return Body().Head(); }
    //: Get head of list.
    
    const DLinkC &Head() const
      { return Body().Head(); }
    //: Get head of list.
    
  public:
    bool IsEmpty() const
      { return Body().IsEmpty(); }
    //: Test is the list is empty.
    
    UIntT Size() const
      { return Body().Size(); }
    //: Count the number of elements in the list.
    
    void Reverse() 
      { return Body().Reverse(); }
    //: Reverse the order of the list.
    
    DListC<DataT> Copy() const
      { return DListC<DataT>(static_cast<DListBodyC<DataT> &>(Body().Copy())); }
    //: Make a copy of this list.
    
    void InsFirst(const DataT &dat)
      { Body().InsFirst(dat); }
    //: Push element onto the begining of the list.
    
    void InsLast(const DataT &dat)
      { Body().InsLast(dat); }
    //: Push element onto the end of the list.
    
    DataT PopFirst()  
      { return Body().PopFirst(); }
    //: Pop item off front of list.
    
    DataT PopLast()  
      { return Body().PopLast(); }
    //: Pop item off end of list.
    
    void DelFirst()
      { Body().PopFirst(); }
    //: Delete the first element from the list.
    
    void DelLast()
      { Body().PopLast(); }
    //: Delete the last element from the list.
    
    void Empty()
      { Body().Empty(); }
    //: Empty the list of all its contents.
    
    DListC<DataT> &operator+=(const DataT &dat)
      { Body().InsLast(dat); return *this; }
    //: Add element to the end of the list.
    
    DListC<DataT> & operator+=(const DListC<DataT> &dat)
      { Body() += dat; return *this; }
    //: Copy contents of other list to end of this one.
    
    void MoveFirst(DListC<DataT> & lst)
      { Body().MoveFirst(lst); }
    //: Move the entire contents of 'lst' to the beginning of this one.
    // this leaves 'lst' empty.
    
    void MoveLast(DListC<DataT> & lst)
      { Body().MoveLast(lst); }
    //: Move the entire contents of 'lst' to the end of this one.
    // this leaves 'lst' empty.

    void MoveFirst(DLIterC<DataT> & at)
      { Body().MoveFirst(at); }
    //: Move the single item 'at' to the beginning of this list.
    
    void MoveLast(DLIterC<DataT> & at)
      { Body().MoveLast(at); }
    //: Move the single item 'at' to the end of this list.
    
    void MergeSort(DListBodyC<DataT>::MergeSortCmpT cmp)
      { Body().MergeSort(cmp); }
    //: Merge sort the list with comparison function 'cmp'.

    bool operator==(const DListC<DataT> &oth) const;
    //: Are lists identical ?
    // Test if lists have identical content.
    
    bool operator!=(const DListC<DataT> &oth) const
      { return !((*this) == oth); }
    //: Are lists different ?
    // Test if lists have different content.
    
    DataT &First() 
      { return Body().First(); }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    DataT &Last() 
      { return  Body().Last(); }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    const DataT &First() const 
      { return Body().First(); }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    const DataT &Last() const 
      { return Body().Last(); }
    //: Get first element in list.
    // NB. List MUST be non-empty.
    
    friend class DLIterC<DataT>;
    friend class DListBodyC<DataT>;

#if RAVL_NEW_ANSI_CXX_DRAFT
    friend ostream &operator<< <DataT>(ostream &strm,const DListC<DataT> &lst);
    friend istream &operator>> <DataT>(istream &strm,DListC<DataT> &lst);
#else
    friend ostream &operator<< (ostream &strm,const DListC<DataT> &lst);
    friend istream &operator>> (istream &strm,DListC<DataT> &lst);
#endif
  };
  
}

#include "Ravl/DLIter.hh"

namespace RavlN {
  
  ///// DListBodyC //////////////////////////////////////////////////////
  
  template<class DataT> 
  RCBodyVC &DListBodyC<DataT>::Copy() const {
    DListBodyC<DataT> *ret = new DListBodyC<DataT>();
    for(DLIterC<DataT> it(*this);it;it++)
      ret->InsLast(*it);
    return *ret;
  }
  
  template<class DataT> 
  void DListBodyC<DataT>::operator+=(const DListC<DataT> &dat) {
    for(DLIterC<DataT> it(dat);it;it++)
      (*this) += *it;
  }

  //: Test if lists are identical.
  // assumes '==' is defined for 'DataT'
  
  template<class DataT> 
  bool DListBodyC<DataT>::operator==(const DListBodyC<DataT> &oth) const {
    DLIterC<DataT> oit(oth),it(*this);
    for(;it && oit;it++,oit++)
      if(!(*it == *oit))
	return false;
    return !(it || oit);
  }
  
  
  template<class DataT>
  ostream &operator<<(ostream &strm,const DListBodyC<DataT> &lst) {
    strm << lst.Size() << "\n";
    for(DLIterC<DataT> it(lst);it;it++)
      strm << *it << "\n";
    return strm;
  }
  //: Send to stream.
  
  template<class DataT>
  istream &operator>>(istream &strm,DListBodyC<DataT> &lst) {
    UIntT i;
    lst.Empty();
    strm >> i;
    for(;i > 0;i--) {
      DataT tmp;
      strm >> tmp;
      lst.InsLast(tmp);
    }
    return strm;
  }
  //: Read from stream.

  ///// DListC //////////////////////////////////////////////////////

  template<class DataT>
  bool DListC<DataT>::operator==(const DListC<DataT> &oth) const {
    if(&Body() == &oth.Body()) // Short cut ?
      return true;
    return Body() == oth.Body();
  }
  
  template<class DataT>
  ostream &operator<<(ostream &strm,const DListC<DataT> &lst) 
  { return strm << lst.Body(); }
  //: Send to stream.
  
  template<class DataT>
  istream &operator>>(istream &strm,DListC<DataT> &lst) {
    DListC<DataT> ret;
    strm >> ret.Body();
    return strm;
  }
  //: Read from stream.
  
}




#endif
