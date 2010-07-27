// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_QUICKSORT_HH
#define	RAVL_QUICKSORT_HH
///////////////////////////////////////////////////////////////////////////
//! file="Ravl/Core/Base/QuickSort.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Charles Galambos"
//! docentry="Ravl.API.Math"
//! date="27/10/2010"

#include "Ravl/Types.hh"

namespace RavlN {

  //! Default less than or equal to used by quick sort.
  template<typename DataT>
  inline bool QuickSortDefaultComparison(const DataT &dat1,const DataT &dat2)
  { return dat1 < dat2; }

  //: Partition contents of array in two.

  template<typename ArrayT, typename IndexT,typename DataT,typename ComparisonFuncT>
  inline IndexT QuickSortPartition(ArrayT &array,
                            const IndexT &start,
                            const IndexT &end,
                            DataT &originalPivotValue,
                            ComparisonFuncT &comparisonFunc)
  {
    DataT &pivotValue = array[end];
    Swap(originalPivotValue,pivotValue);
    IndexT at = start;
    for(IndexT i = start; i < end; ++i) {
      if(comparisonFunc(array[i],pivotValue)) {
        Swap(array[at], array[i]);
        ++at;
      }
    }
    Swap(pivotValue, array[at]);
    return at;
  }

  //! Quick sort contents of array between start and end inclusive.

  template<typename ArrayT, typename IndexT,typename ComparisonFuncT>
  void QuickSort(ArrayT &array, const IndexT &start, const IndexT &end,ComparisonFuncT &compFunc = ComparisonFuncT())
  {
    if(start >= end)
      return ; // One element or less, no sorting needed!
    IndexT pivot = (start + end) / 2;
    // Compute the median of first, middle and last to choose the pivot.
    if(compFunc(array[start],array[pivot]) && compFunc(array[end],array[start])) {
      pivot=start;
    } else {
      if(compFunc(array[end],array[pivot]) && compFunc(array[start],array[end]))
        pivot=end;
    }
    pivot = QuickSortPartition(array,start, end,array[pivot],compFunc);
    QuickSort(array,start, pivot-1,compFunc);
    QuickSort(array,pivot + 1, end,compFunc);
  }

  //! Helper method where no default comparison is provided.
  template<typename ArrayT, typename IndexT, typename DataT>
  inline void QuickSortTyped(ArrayT &array, const IndexT &start,const IndexT &end,const DataT &value)
  { return QuickSort(array,start,end,QuickSortDefaultComparison<DataT>); }

  //! Quick sort using the default comparison operator <=
  
  template<typename ArrayT, typename IndexT>
  inline void QuickSort(ArrayT &array, const IndexT &start, const IndexT &end)
  { return QuickSortTyped(array,start,end,array[start]); }

}


#endif	/* QUICKSORT_HH */

