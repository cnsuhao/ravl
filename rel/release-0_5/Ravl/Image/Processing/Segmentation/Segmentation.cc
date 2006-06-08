// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlImage
//! file="Ravl/Image/Processing/Segmentation/Segmentation.cc"

#include "Ravl/Image/Segmentation.hh"
#include "Ravl/Array2dSqr2Iter.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/SArr1Iter.hh"
#include "Ravl/SArr1Iter2.hh"
#include "Ravl/Array2dIter.hh"

namespace RavlImageN {

  //: Generate a table of region adjacencies.
  
  SArray1dC<HSetC<UIntT> > SegmentationBodyC::Adjacency(bool biDir) {
    SArray1dC<HSetC<UIntT> > ret(labels);
    if(biDir) {
      for(Array2dSqr2IterC<UIntT> it(segmap);it;) {
	if(it.DataBR() != it.DataTR()) {
	  ret[it.DataBR()] += it.DataTR();
	  ret[it.DataTR()] += it.DataBR();
	}
	for(;it.Next();) { // The rest of the image row.
	  if(it.DataBR() != it.DataTR()) {
	    ret[it.DataBR()] += it.DataTR();
	    ret[it.DataTR()] += it.DataBR();
	  }
	  if(it.DataBR() != it.DataBL()) {
	    ret[it.DataBR()] += it.DataBL();
	    ret[it.DataBL()] += it.DataBR();
	  }
	}
      }
    } else {
      for(Array2dSqr2IterC<UIntT> it(segmap);it;) {
	if(it.DataBR() != it.DataTR()) {
	  if(it.DataBR() < it.DataTR())
	    ret[it.DataBR()] += it.DataTR();
	  else
	    ret[it.DataTR()] += it.DataBR();
	}
	for(;it.Next();) { // The rest of the image row.
	  if(it.DataBR() != it.DataTR()) {
	    if(it.DataBR() < it.DataTR())
	      ret[it.DataBR()] += it.DataTR();
	    else
	      ret[it.DataTR()] += it.DataBR();
	  }
	  if(it.DataBR() != it.DataBL()) {
	    if(it.DataBR() < it.DataBL())
	      ret[it.DataBR()] += it.DataBL();
	    else
	      ret[it.DataBL()] += it.DataBR();
	  }
	}
      }
    }
    return ret;
  }
  
  //: Generate a table of region adjacencies boundry lengths.
  // only adjacenies from regions with a smaller id to those 
  // with a larger ID are generated
  
  SArray1dC<HashC<UIntT,UIntC> > SegmentationBodyC::BoundryLength() {
    SArray1dC<HashC<UIntT,UIntC> > ret(labels);
    for(Array2dSqr2IterC<UIntT> it(segmap);it;) {
      if(it.DataBR() != it.DataTR()) {
	if(it.DataBR() < it.DataTR())
	  ret[it.DataBR()][it.DataTR()]++;
	else
	  ret[it.DataTR()][it.DataBR()]++;
      }
      for(;it.Next();) { // The rest of the image row.
	if(it.DataBR() != it.DataTR()) {
	  if(it.DataBR() < it.DataTR())
	    ret[it.DataBR()][it.DataTR()]++;
	  else
	    ret[it.DataTR()][it.DataBR()]++;
	}
	if(it.DataBR() != it.DataBL()) {
	  if(it.DataBR() < it.DataBL())
	    ret[it.DataBR()][it.DataBL()]++;
	  else
	    ret[it.DataBL()][it.DataBR()]++;
	}
      }
    }
    return ret;
  }
  
  
  //: recompute the areas from the original areas and a mapping.
  
  SArray1dC<UIntT> SegmentationBodyC::RedoArea(SArray1dC<UIntT> area,SArray1dC<UIntT> map) {
    SArray1dC<UIntT> ret(labels);
    ret.Fill(0);
    for(SArray1dIter2C<UIntT,UIntT> it(area,map);it;it++)
      ret[it.Data2()] += it.Data1();
    return ret;
  }
  
  //: Compute the areas of all the segmented regions.
  
  SArray1dC<UIntT> SegmentationBodyC::Areas() {
    // Compute areas of components
    SArray1dC<UIntT> area(labels);
    area.Fill(0);  // Initilisation
    for(Array2dIterC<UIntT> it(segmap);it;it++)
      area[*it]++;
    return area;
  }
  
  //: Make an array of labels mapping to themselves.
  
  SArray1dC<UIntT> SegmentationBodyC::IdentityLabel() {
    // Make an identity mapping.
    SArray1dC<UIntT> minLab(labels);
    UIntT c = 0;
    for(SArray1dIterC<UIntT> it(minLab);it;it++)
      *it = c++;
    return minLab;
  }
  
//: Compress labels.
  
  UIntT SegmentationBodyC::RelabelTable(SArray1dC<UIntT> &labelTable, UIntT currentMaxLabel) 
    // The 'labelTable' represents a look-up table for labels. 
    // Each item contains a new label which can be the same
    // as the index of the item or smaller. Such 'labelTable' contains
    // a forest of labels, every tree of labels represents one component
    // which should have the same label. It is valid that a root item
    // of a tree has the same label value as the item index.
  {
    // Make all trees of labels to be with depth one.
    for(SArray1dIterC<UIntT> it(labelTable,currentMaxLabel+1);it;it++)
      *it = labelTable[*it];
    
    // Now all components in the 'labelTable' have a unique label.
    // But there can exist holes in the sequence of labels.
    
    // Squeeze the table. 
    UIntT n = 0;                     // the next new label  
    for(SArray1dIterC<UIntT> it(labelTable,currentMaxLabel+1);it;it++) {
      UIntT m = labelTable[*it];  // the label of the tree root
      
      // In the case m >= n the item with the index 'l' contains 
      // the root of the new tree,
      // because all processed roots have a label smaller than 'n'.
      // The root label 'm' has the same value as the index 'l'.
      // The root will be relabeled by a new label.
      *it = (m >= n) ? n++ : m;
    }
    
    return n - 1;  // the new last used label
  }
  
  //: Compress newlabs and re-label segmentation.
  
  UIntT SegmentationBodyC::CompressAndRelabel(SArray1dC<UIntT> newLabs) {
    // ------ Compress labels -----
    // First make sure they form a directed tree
    // ending in the lowest valued label. 
    UIntT n = 0;
    for(SArray1dIterC<UIntT> it(newLabs);it;it++,n++) {
      if(*it > n) {
	// Minimize label.
	UIntT nat,at = *it;
	UIntT min = n;
	for(;at > n;at = nat) {
	  nat = newLabs[at];
	  if(nat < min)
	    min = nat;
	  else
	    newLabs[at] = min;	
	}
	*it = min;
      }
    }
    
    // Now we can minimize the labels.
    UIntT newLastLabel = RelabelTable(newLabs,labels-1);
    
    // And relable the image.
    for(Array2dIterC<UIntT> it(segmap);it;it++)
      *it = newLabs[*it];
    
    labels = newLastLabel+1;
    return labels;
  }
  
  
  //: Remove small components from map, label them as 0.
  
  UIntT SegmentationBodyC::RemoveSmallComponents(IntT thrSize) {
    if(labels <= 1)
      return labels;
    
    SArray1dC<UIntT> area = Areas();
    
    // Assign new labels to the regions according their sizes and
    // another requirements.
    IntT newLabel = 1;
    SArray1dIterC<UIntT> it(area);
    *it = 0;
    it.Next();
    for(;it;it++) {
      if (*it < ((UIntT) thrSize)) 
	*it = 0;
      else 
	*it = newLabel++;
    }
    
    // Remove small components
    for(Array2dIterC<UIntT> it(segmap);it;it++)
      *it = area[*it];
    
    labels = newLabel;
    return newLabel;
    
  }
  
}