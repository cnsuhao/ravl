// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlCore

#include "Ravl/PriQueueL.hh"
#include "Ravl/EntryPnt.hh"
#include <iostream.h>

using namespace RavlN;

int Main(int, char*[])
{
  PriQueueLC<IntT, IntT> pq;

  pq.Insert(11,1);
  pq.Insert( 5,2);
  pq.Insert( 6,3);
  pq.Insert( 3,4);
  pq.Insert( 1,5);
  pq.Insert( 4,6);
  pq.Insert( 2,7);
  pq.Insert( 7,8);
  pq.Insert(10,9);
  pq.Insert( 9,10);
  pq.Insert( 8,11);
  pq.Insert(12,12);

  int count = 12;
  int lastkey = -1;
  while (pq.IsElm()){
    if(lastkey > pq.TopKey()) {
      cerr << "Failed Data out of order. \n";
      return 1;
    }
    lastkey = pq.TopKey();
#if 0
    pq.Dump();
    cerr << "pq.Size() = " << pq.Size() << "\n";
    cerr << "extracting Key:" << pq.TopKey() << " "<< pq.Top() << "\n";
    cerr << "iteration " << count << "\n\n";
#endif
    count--;
    pq.DelTop();
  }
  if(count != 0) {
    cerr << "Count mis-match. " << count << "\n";
    return 1;
  }
  cerr << "Test passed ok. \n";
  return 0;
}

RAVL_ENTRY_POINT(Main);

