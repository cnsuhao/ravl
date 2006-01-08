// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImageProc
//! author="Charles Galambos, based on code by Jiri Matas."
//! file="Ravl/Image/Processing/Segmentation/SegmentExtrema.cc"

#include "Ravl/Image/SegmentExtrema.hh"
#include "Ravl/Image/DrawFrame.hh"
#include "Ravl/Array2dIter2.hh"
#include "Ravl/Array1dIter.hh"
#include "Ravl/IO.hh"
#include <string.h>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {

  //: Destructor.
  
  SegmentExtremaBaseC::~SegmentExtremaBaseC() {
    ReallocRegionMap(0);
  }

  //: Delete the current region set, free any memory used.
  
  void SegmentExtremaBaseC::ReallocRegionMap(IntT size) {
    if(size == 0)
      regionMap = SArray1dC<ExtremaRegionC>();
    else 
      regionMap = SArray1dC<ExtremaRegionC>(size);
  }
  
  //: Setup structures for a given image size.
  
  void SegmentExtremaBaseC::SetupImage(const IndexRange2dC &rect) {
    IndexRange2dC imgRect = rect.Expand(1);
    labelAlloc = 0; // Reset region allocation counter.
    if(imgRect == pixs.Frame())
      return ; // Done already.
    
    // Allocate ExtremaChainPixelC image.
    pixs = ImageC<ExtremaChainPixelC>(imgRect);
    origin = &(pixs[rect.Origin()]);
    stride = imgRect.Cols();
    
    // Put a frame of zero labels around the edge.
    ExtremaChainPixelC zeroPix;
    zeroPix.region = 0;
    zeroPix.next = 0;
    DrawFrame(pixs,zeroPix,imgRect);
    
    // Create region map.
    ReallocRegionMap(rect.Area()/2);
    
    // ...
    maxSize = rect.Area();
  }
  
  //: Find matching label.
  inline
  ExtremaRegionC * SegmentExtremaBaseC::FindLabel(ExtremaChainPixelC *cp) {
    register ExtremaRegionC *lab = cp->region;
    if(lab == 0 || lab->merge == 0) return lab;
    register ExtremaRegionC *at = lab->merge;
    while(at->merge != 0)
      at = at->merge;
    // Relabel mappings.
    do {
      register ExtremaRegionC *tmp = lab;
      lab = lab->merge;
      tmp->merge = at;
    } while(lab != at);
    cp->region = at; // This makes things run slightly faster, but we loose info about the pixel.
    return at;
  }

  //: Find the labels around the pixel 'pix'
  // put the results into 'labelArray' which must be at least 4 labels long.
  // The number of labels found is returned.
  inline
  int SegmentExtremaBaseC::ConnectedLabels(ExtremaChainPixelC *pix,ExtremaRegionC **labelArray) {
    //cerr << "SegmentExtremaBaseC::ConnectedLabels(), Pix=" << ((void *) pix) << "\n";
    IntT n = 0;
    ExtremaRegionC *l1 = FindLabel(pix + 1);
    if (l1!=0 )                               { labelArray[n++]=l1; }
    ExtremaRegionC *l2 = FindLabel(pix - 1);
    if (l2!=0 && l2!=l1)                      { labelArray[n++]=l2; }
    ExtremaRegionC *l3 = FindLabel(pix + stride);
    if (l3!=0 && l3!=l1 && l3!=l2)            { labelArray[n++]=l3; }
    ExtremaRegionC *l4 = FindLabel(pix - stride);
    if (l4!=0 && l4!=l1 && l4!=l2 && l4!=l3)  { labelArray[n++]=l4; }
    return n;
  }

  //: Find the labels around the pixel 'pix'
  // put the results into 'labelArray' which must be at least 4 labels long.
  // The number of labels found is returned.
  inline
  int SegmentExtremaBaseC::ConnectedLabels6(ExtremaChainPixelC *pix,ExtremaRegionC **labelArray) {
    //cerr << "SegmentExtremaBaseC::ConnectedLabels(), Pix=" << ((void *) pix) << "\n";
    IntT n = 0;
    ExtremaRegionC *l1 = FindLabel(pix + 1);
    if (l1!=0 )                               { labelArray[n++]=l1; }
    ExtremaRegionC *l2 = FindLabel(pix - 2);
    if (l2!=0 && l2!=l1)                      { labelArray[n++]=l2; }
    ExtremaRegionC *l3 = FindLabel(pix + stride);
    if (l3!=0 && l3!=l1 && l3!=l2)            { labelArray[n++]=l3; }
    ExtremaRegionC *l4 = FindLabel(pix - stride);
    if (l4!=0 && l4!=l1 && l4!=l2 && l4!=l3)  { labelArray[n++]=l4; }
    ExtremaRegionC *l5 = FindLabel(pix + stride -1);
    if (l5!=0 && l5!=l1 && l5!=l2 && l5!=l3 && l5!=l4)           { labelArray[n++]=l5; }
    ExtremaRegionC *l6 = FindLabel(pix - stride -1);
    if (l6!=0 && l6!=l1 && l6!=l2 && l6!=l3 && l6!=l4 && l6!=l5)  { labelArray[n++]=l6; }
    return n;
  }
  
  //: Add a new region.
  inline
  void SegmentExtremaBaseC::AddRegion(ExtremaChainPixelC *pix,IntT level) {
    ExtremaRegionC &region = regionMap[labelAlloc++];
    pix->region = &region;
    //cerr << "SegmentExtremaBaseC::AddRegion(), Pix=" << (void *) pix << " Region=" << (void *) &region << "\n";
    IntT offset = pix - origin;
    region.total = 0;
    region.merge = 0; //&region;
    IntT nlevel = level+1; // Don't need to clear this level as its going to be set anyway
    if(region.hist == 0) {
      region.hist = PopHist(nlevel);
    } else {
      IntT clearSize = (region.maxValue + 1) - nlevel;
      if(clearSize > 0)
	memset(&(region.hist[nlevel]),0,clearSize * sizeof(IntT));
    }
#if 0
    // Check the histogram is clear.
    for(int i = level;i <= valueRange.Max().V();i++) {
      if(region.hist[i] != 0) 
        cerr << "Non zero at " << i << " Max=" << valueRange.Max().V() <<  "\n";
      RavlAlwaysAssert(region.hist[i] == 0);
    }
#endif

    region.minValue = level;
    region.maxValue = valueRange.Max().V();
    region.minat = Index2dC((offset / stride),(offset % stride)) + pixs.Frame().Origin() + Index2dC(1,1);
    region.hist[level] = 1;
    region.total = 1;
    region.thresh = 0;
    region.closed = 0;
  }
  
  //: Add pixel to region.
  
  inline
  void SegmentExtremaBaseC::AddPixel(ExtremaChainPixelC *pix,IntT level,ExtremaRegionC *reg) {
    reg->hist[level]++;
    reg->total++;
    pix->region = reg;
  }

  //: Add pixel to region.
  
  inline
  void SegmentExtremaBaseC::MergeRegions(ExtremaChainPixelC *pix,IntT level,ExtremaRegionC **labels,IntT n) {
    ExtremaRegionC *max = labels[0];
    IntT maxValue = labels[0]->total;
    
    // Find largest region.
    int i;
    for(i = 1;i < n;i++) {
      IntT tot = labels[i]->total;
      if(maxValue < tot) {
	max = labels[i];
	maxValue = tot;
      }
    }
    
    // Merge regions, and finalise
    ExtremaRegionC &nr = *max;
    for(i = 0;i < n;i++) {
      if(labels[i] == max)
	continue;
      ExtremaRegionC &oldr = *labels[i];
      oldr.maxValue = level;
      oldr.merge = max;
      oldr.closed = max;
      
      // If we don't need the histogram, put it back in a pool
      if(oldr.hist != 0 && (oldr.total < minSize || (level - oldr.minValue) < minMargin) ) {
	PushHist(oldr.hist,level);
	oldr.hist = 0;
      }
      
      nr.total += oldr.total;
      nr.hist[level] += oldr.total;
    }
    AddPixel(pix,level,max);
  }
  
  //: Generate regions.
  
  void SegmentExtremaBaseC::GenerateRegions() {
    ExtremaChainPixelC *at;
    IntT n, clevel = levels.Range().Min().V();
    ExtremaRegionC *labels[6];
    for(Array1dIterC<ExtremaChainPixelC *> lit(levels);lit;lit++,clevel++) {
      //ONDEBUG(cerr << "Level=" << clevel << " \n");
      for(at = *lit;at != 0;at = at->next) {
#if 0
	// Is the next pixel the same level ?
	if(at->next != (at-1)) {
#endif
	  // Got a standard pixel.
	  //ONDEBUG(cerr << "Pixel=" << (void *) at << " Region=" << at->region << "\n");
	  n = ConnectedLabels(at,labels);
	  switch(n) {
	  case 0: AddRegion(at,clevel);  break;
	  case 1: AddPixel(at,clevel,labels[0]); break;
	  default: MergeRegions(at,clevel,labels,n); break;
	  }
#if 0
	} else {
	  // Got two pixels in a row.
	  n = ConnectedLabels6(at,labels);
	  switch(n) {
	  case 0: AddRegion(at,clevel);  break;
	  case 1: AddPixel(at,clevel,labels[0]); break;
	  default: MergeRegions(at,clevel,labels,n); break;
	  }
	  // Add in second pixel.
	  AddPixel(at->next,clevel,at->region);
	  at = at->next; // Next pixel is done as well
	}
#endif
      }
    }
    //cerr << "Regions labeled = " << labelAlloc << "\n";
  }
  
  //: Generate thresholds
  
  void SegmentExtremaBaseC::Thresholds() {
    //cerr << "SegmentExtremaBaseC::Thresholds() ********************************************** \n";
    SArray1dC<ExtremaThresholdC> thresh(limitMaxValue + 2);
    IntT nthresh;
    Array1dC<IntT> chist(0,limitMaxValue + 2);
    chist.Fill(0);
    int half_perimeter_i;
    
    for(SArray1dIterC<ExtremaRegionC> it(regionMap,labelAlloc);it;it++) {
      if(it->total < minSize || (it->maxValue - it->minValue) < minMargin) {
	it->nThresh = 0;// Ingore these regions.
	continue; // Not enough levels in the region.
      }
      // Build the cumulative histogram.
      IntT maxValue = it->maxValue;
      IntT minValue = it->minValue;
      IntT sum = 0;
      IntT i;
      
      ONDEBUG(cerr << "Hist= " << it->minValue << " :");
      for(i = minValue;i <= maxValue;i++) {
	sum += it->hist[i];
	chist[i] = sum;
	ONDEBUG(cerr << " " << it->hist[i]);
      }
      chist[maxValue] = sum;
      
      ONDEBUG(cerr << "  Closed=" << (it->closed != 0)<< "\n");
      IntT up;
      // look for threshold that guarantee area bigger than minSize.
      for(i = minValue; i <= maxValue;i++)
	if(chist[i] >= minSize) break; 
      // Find thresholds.
      nthresh = 0;
      ONDEBUG(cerr << "Min=" << minValue << " Max=" << maxValue << " Init=" << i << " MaxSize=" << maxSize << "\n");
      IntT lastThresh = 0;
      for(up=i+1; up < maxValue && i < maxValue; i++) {
	int area_i = chist[i];
	if(area_i > maxSize) {
	  //cerr << "Size limit reached. \n";
	  break; // Quit if area is too large.
	}
        
	half_perimeter_i = Round(2.1 * Sqrt((double)area_i)) + area_i;
        if(half_perimeter_i == lastThresh)
          continue; // If the thresholds are the same, the next margin will only be shorter.
        lastThresh = half_perimeter_i;
	while(up <= maxValue && chist[up] < half_perimeter_i)
	  up++;
	
	int margin = up - i;
	//ONDEBUG(cerr << " Margin=" << margin << "\n");
	if(margin > minMargin) { // && margin > prevMargin
	  ExtremaThresholdC &et = thresh[nthresh++];
	  et.pos = i;
 	  et.margin = margin;
	  et.thresh = i + margin/2;
	  // cerr << " Threshold=" << et.thresh << " " << chist[et.thresh] << "\n";
	}
      }
      
      if(it->closed == 0) { // Is region closed ?
	ExtremaThresholdC &et = thresh[nthresh++];
	et.pos = maxValue-1;
	int margin = up - i;
	et.margin = margin;
	et.thresh = maxValue-1;	
      }
      //ONDEBUG(cerr << "NThresh=" <<  nthresh << "\n");
      ExtremaThresholdC *newthresh = new ExtremaThresholdC[nthresh];
      IntT nt = 0;
      
      IntT lastSize = -1;
      IntT lastInd = -1;
      for(int j = 0;j < nthresh;j++) {
	UIntT size = chist[thresh[j].pos];
	if((lastSize * 1.15) > size) { // Is size only slighly different ?
	  if(thresh[j].margin > thresh[lastInd].margin) {
	    newthresh[nt-1] = thresh[j]; // Move threshold if margin is bigger.
            lastSize = size;
          }
	  continue; // Reject it, not enough difference.
	}
	newthresh[nt++] = thresh[j];
	lastSize = size;
	lastInd = j;
      }
      
      if(nt > 0) {
	it->thresh = newthresh;
	it->nThresh = nt;
      } else {
	it->nThresh = 0;
	delete [] newthresh;
      }
      //cerr << "Thresholds=" << nthresh << " Kept=" << it->nThresh << "\n";
    } // for(SArray1dIterC<ExtremaRegionC> it(...
    //cerr << "SegmentExtremaBaseC::Thresholds() Interesting regions=" << regions <<" \n";
  }  
}
