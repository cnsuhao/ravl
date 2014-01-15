// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Container/Queue/testQueue.cc"
//! userlevel=Develop
//! author="Charles Galambos"
//! docentry="Ravl.API.Core.Queues"

#include "Ravl/BlkQueue.hh"
#include "Ravl/Stream.hh"
#include "Ravl/UnitTest.hh"
#include <stdlib.h>

using namespace RavlN;

int testQueue();

int main() {

  RAVL_RUN_TEST(testQueue());

  std::cerr << "Queue test passed ok. \n";

  return 0;
}

int testQueue()
{
  BlkQueueC<int> q;
  int count = 0;
  int lget = -1;
  for(int i = 0;i < 5000;i++) {
    switch(rand() % 2)
      {
      case 0:
        q.InsLast(count++);
        break;
      case 1:
        if(q.IsEmpty())
          continue;
        {
          int lval = q.GetFirst();
          if(lval != (lget+1)) {
            std::cerr << "Queue test failed. \n";
            return __LINE__;
          }
          lget = lval;
        }
        break;
      }
  }

  return 0;
}

template class BlkQueueC<int>;
