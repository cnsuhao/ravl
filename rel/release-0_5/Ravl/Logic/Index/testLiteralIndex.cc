// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsi="$Id$"
//! rcsid="$Id$"
//! lib=RavlLogic
//! file="Ravl/Logic/Index/testLiteralIndex.cc"

#include "Ravl/Logic/LiteralIndex.hh"
#include "Ravl/Logic/Tuple.hh"
#include "Ravl/Logic/LiteralIndexFilter.hh"

using namespace RavlLogicN;

IntT testBaseTest();
IntT testIndexFilterTest();

int main()
{
  UIntT err;
  
  if((err = testBaseTest()) != 0) {
    cerr << "LiteralIndex test failed, line: " << err << " \n";
    return 1;
  }
  if((err = testIndexFilterTest()) != 0) {
    cerr << "LiteralIndexFilter test failed, line: " << err << " \n";
    return 1;
  }
  
  cerr << "LiteralIndex test passed. \n";
  return 0;
}

IntT testBaseTest() {
  LiteralC l1 = Literal();
  LiteralC l2 = Literal();
  LiteralC t1 = Tuple(l1);
  LiteralC t2 = Tuple(l2,l1);
  LiteralC t3 = Tuple(l1,l1);
  
  LiteralIndexC<UIntT> index(true);
  //cerr << "\nSetting l1\n";
  index[l1] = 0; 
  //cerr << "\nSetting t1\n";
  index[t1] = 1; 
  //cerr << "\nSetting t2\n";
  index[t2] = 2;
  //cerr << "\nSetting t3\n";
  index[t3] = 3; 
  if(index[l1] != 0) return __LINE__;
  if(index[t1] != 1) return __LINE__;
  if(index[t2] != 2) return __LINE__;
  if(index[t3] != 3) return __LINE__;
  if(index.Size() != 4) return  __LINE__;
  //  cerr << "Dump:\n";
  //index.Dump(cerr);
  return 0;
}

IntT testIndexFilterTest() {
  cerr << "testIndexFilterTest() \n";
  LiteralIndexC<UIntT> index(true);
  LiteralC l1 = Literal();
  LiteralC l2 = Literal();
  LiteralC l3 = Literal();
  LiteralC t1 = Tuple(l1);
  LiteralC t2 = Tuple(l2,l1);
  LiteralC t3 = Tuple(l1,l1);
  LiteralC t4 = Tuple(l1,l3);
  
  VarC v1 = Var();
  
  cerr << "v1: " << v1 << "\n";
  cerr << "l1: " << l1 << "\n";
  
  index[l1] = 0; 
  index[t1] = 1; 
  index[t2] = 2;
  index[t3] = 3; 
  index[t4] = 4; 
  UIntT count = 0;
  
  // Test filter on variable.
  cerr << "--- Filter on var. \n";
  for(LiteralIndexFilterC<UIntT> it(index,v1);it;it++)
    count++;
  if(count != 5) return __LINE__;
  
  cerr << "--- Filter on tuple (v1,l1) \n";
  
  // Test filter on tuple with single var.
  count = 0;
  LiteralC f1 = Tuple(v1,l1);
  for(LiteralIndexFilterC<UIntT> it(index,f1);it;it++) {
    count++;
    if(it.MappedData() == 2)
      continue;
    if(it.MappedData() == 3)
      continue;
    cerr << "Filter failed :" << it.Data().Name() << " Value:" << it.MappedData() << "\n";
    return __LINE__;
  }
  if(count != 2) return __LINE__;
  count = 0;
  
  // Test filter on another tuple with single var.
  cerr << "--- Filter on tuple (l2,v1) \n";
  LiteralC f2 = Tuple(l2,v1);
  for(LiteralIndexFilterC<UIntT> it(index,f2);it;it++) {
    count++;
    if(it.MappedData() == 2)
      continue;
    cerr << "Filter failed :" << it.Data().Name() << " Value:" << it.MappedData() << "\n";
    return __LINE__;
  }
  if(count != 1) return __LINE__;
  
  // Test multi variable filter.
  cerr << "--- Filter on tuple (v1,v1) \n";
  count = 0;
  LiteralC f3 = Tuple(v1,v1);
  for(LiteralIndexFilterC<UIntT> it(index,f3);it;it++) {
    count++;
    if(it.MappedData() == 3)
      continue;
    cerr << "Filter failed :" << it.Data().Name() << " Value:" << it.MappedData() << "\n";
    return __LINE__;
  }
  if(count != 1) return __LINE__;
  
  return 0;
}