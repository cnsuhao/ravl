#ifndef RAVLBINTABLET_HEADER
#define RAVLBINTABLET_HEADER 1
///////////////////////////////////////////////////////////////
//! file="amma/Statist/SparHist/BinTable.hh"
//! userlevel=Normal
//! author="Charles Galambos"
//! date="27/11/1996"
//! docentry="Ravl.Core.Misc"
//! rcsid="$Id$"
//! lib=SparHist

#include "Ravl/Hash.hh"
#include "Ravl/Math.hh"

namespace RavlN {

  template<class IT,class DIT,class BT> class BinIterC;
  template<class IT,class DIT,class BT> class BinTableC;
  template<class IT,class DIT,class BT> ostream &operator<<(ostream &s,const BinTableC<IT,DIT,BT> &);
  template<class IT,class DIT,class BT> istream &operator>>(istream &s,BinTableC<IT,DIT,BT> &);
  
  /////////////////////////////////
  //: A table of bins.  Hashing for real valued paramiters.
  // Note: This only works for multi-dementional tables.
  //
  // SMALL OBJECT <p>
  // Template paramiter.            Examples. <p>
  //
  // IT  - Index Type.              ( Vector2dC ) <p>
  // DIT - Descritised index type.  ( Index2dC  ) <p>
  // BT  - Bin Type.                (     x     ) <p>
  // 
  
  template<class IT,class DIT,class BT>
  class BinTableC {
  public:  
    inline BinTableC(const IT &nBinSize)
      : binSize(nBinSize) 
      {}
    //: Constructor.
    
    inline BT &operator[](const IT &Pnt)
      { return bins[Scale(Pnt)]; }
    //: Access a bin.
    
    inline const BT *operator[](const IT &Pnt) const;
    //: Constant access to a bin.
    // May return NULL !
    
    BT &Bin(const IT &Pnt)
      { return bins[Scale(Pnt)]; }
    //: Access a bin.
    
    BT &Bin(const DIT &Pnt) 
      { return bins[Pnt]; }
    //: Access a bin directly.
    
    //inline const BT *Bin(const IT &Pnt) const;
    // Constant access to a bin.
    // May return NULL !
    
    void Empty(void) { bins.Empty(); }
    //: Remove all items from table.
    
    const IT &BinSize(void) const { return binSize; }
    //: Get size of bins.
    
    DIT Scale(const IT &Ind) const;
    //: Scale and descritise an input point.
    
    BT *GetBin(const IT &Pnt) { return bins.Lookup(Scale(Pnt)); }
    //: See if bins present.
    
    IT BinCentre(const IT &at) const;
    //: Get centre of bin containing point at.
    
    bool IsEmpty() const
      { return bins.IsEmpty(); }
    //: Test if table is empty.
    
    void SetBinSize(const IT &nBinSize) { 
      RavlAssert(IsEmpty());
      binSize = nBinSize; 
    }
    //: Set the bin size.  
    // NB. This can only be used with an empty table.
  private:
    IT binSize;    // Scaling factor for bins.  
    HashC<DIT,BT> bins; // Table of bins.
    
    friend class BinIterC<IT,DIT,BT>;
    
#if !defined(__sgi__) && !defined(VISUAL_CPP)
    friend ostream &operator<< <>(ostream &s,const BinTableC<IT,DIT,BT> &);
    friend istream &operator>> <>(istream &s,BinTableC<IT,DIT,BT> &);
#else
    friend ostream &operator<<(ostream &s,const BinTableC<IT,DIT,BT> &);
    friend istream &operator>>(istream &s,BinTableC<IT,DIT,BT> &);
#endif
  };
  
  //////////////////////////
  
  template<class IT,class DIT,class BT>
  inline DIT BinTableC<IT,DIT,BT>::Scale(const IT &Ind) const  {
    DIT Ret(Ind.Size());
    for(UIntT i = 0;i < Ind.Size();i++) 
      Ret[i] = Floor((RealT) Ind[i] / binSize[i]);
    return Ret;
  }  
  
  template<class IT,class DIT,class BT>
  inline const BT * BinTableC<IT,DIT,BT>::operator[](const IT &Pnt) const  {
    const BT *Ptr = bins.Lookup(Scale(Pnt));
    if(Ptr == 0)
      return 0;
    return Ptr;
  }
  
  template<class IT,class DIT,class BT>
  IT BinTableC<IT,DIT,BT>::BinCentre(const IT &at) const {
    IT ret(at.Size());
    for(UIntT i = 0;i < at.Size();i++) 
      ret[i] = Floor((RealT) at[i] / binSize[i]) * binSize[i]  + (binSize[i]/2);
    return ret;
  }
  
  template<class IT,class DIT,class BT>
  ostream &operator<<(ostream &s,const BinTableC<IT,DIT,BT> &tab) {
    s << tab.BinSize() << ' ' << tab.bins;
    return s;
  }
  //: Write bin table to a stream.
  
  template<class IT,class DIT,class BT>
  istream &operator>>(istream &s,BinTableC<IT,DIT,BT> &tab) {
    s >> tab.binSize >> tab.bins;
    return s;
  }
  //: Read bin table from a stream.
  
}

#endif
