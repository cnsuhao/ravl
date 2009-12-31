// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2009, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_UNITTEST_HEADER
#define RAVL_UNITTEST_HEADER 1
/////////////////////////////////////////////////////////////////////////
//! docentry="Ravl.API.Core"
//! file="Ravl/Core/Base/UnitTest.hh"
//! lib=RavlCore
//! userlevel=Normal
//! author="Charles Galambos"

#include "Ravl/BinStream.hh"
#include "Ravl/StrStream.hh"

// Helper functions for unit testing RAVL code.
namespace RavlN {

  //! Test binary IO.
  // Its up to the implementer to check the reloaded
  // value is equal to the original input.
  // Returns true if stream is synchronised on output.

  template<typename DataT>
  bool TestBinStreamIO(const DataT &input,DataT &reloaded) {
    // Write data out.
    StringC strmData;
    UIntT checkValue = 0x12345678;
    {
      StrOStreamC ostrm;
      BinOStreamC bos(ostrm);
      bos << input << checkValue;
      strmData = ostrm.String();
    }

    // Read data in.
    {
      StrIStreamC strm(strmData);
      BinIStreamC bis(strm);
      UIntT loadedCheckValue = 0;
      bis >> reloaded >> loadedCheckValue;

      // Check the stream length matches.
      if(loadedCheckValue != checkValue) return false;
    }

    return true;
  }

  template<typename DataT>
  bool TestEquals(const DataT &expected, const DataT &actual, const char* file, int line) {
    if (!(expected == actual)) {
      std::cerr << file << ":" << line << " Expected value (" << expected << ") but instead got (" << actual << ")\n";
      return false;
    }
    return true;
  }

  template<typename DataT>
  bool TestNotEquals(const DataT &expected, const DataT &actual, const char* file, int line) {
    if (expected == actual) {
      std::cerr << file << ":" << line << " Value should not equal (" << expected << ")\n";
      return false;
    }
    return true;
  }

  template<typename DataT>
  bool TestTrue(const DataT &value, const char* file, int line) {
    if (!value) {
      std::cerr << file << ":" << line << " Expression should be true\n";
      return false;
    }
    return true;
  }

  template<typename DataT>
  bool TestFalse(const DataT &value, const char* file, int line) {
    if (value) {
      std::cerr << file << ":" << line << " Expression should be false\n";
      return false;
    }
    return true;
  }
}

#define RAVL_TEST_EQUALS(x,y) { \
  if (!RavlN::TestEquals((x), (y), __FILE__, __LINE__)) \
    return __LINE__; \
  }

#define RAVL_TEST_NOT_EQUALS(x,y) { \
  if (!RavlN::TestNotEquals((x), (y), __FILE__, __LINE__)) \
    return __LINE__; \
  }

#define RAVL_TEST_TRUE(x) { \
  if (!RavlN::TestTrue((x), __FILE__, __LINE__)) \
    return __LINE__; \
  }

#define RAVL_TEST_FALSE(x) { \
  if (!RavlN::TestFalse((x), __FILE__, __LINE__)) \
    return __LINE__; \
  }

#endif
