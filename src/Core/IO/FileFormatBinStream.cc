// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////////
//! lib=RavlIO
//! file="Ravl/Core/IO/FileFormatBinStream.cc"

#include "Ravl/DP/FileFormatBinStream.hh"

namespace RavlN {
  
  void InitFileFormatBinStream() 
  {}
  
  // Some basic file types.
  
  static FileFormatBinStreamC<RealT>   FileFormatBinStream_RealT;
  static FileFormatBinStreamC<IntT>    FileFormatBinStream_IntT;
  static FileFormatBinStreamC<UIntT>   FileFormatBinStream_UIntT;
  static FileFormatBinStreamC<StringC> FileFormatBinStream_StringC;
  static FileFormatBinStreamC<ByteT>   FileFormatBinStream_ByteT;
}
