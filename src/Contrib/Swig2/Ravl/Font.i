// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

%include "Ravl/Swig2/Image.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/Image/Font.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif

// The following is a fix for windows.
#define NODRAWTEXT
%}

namespace RavlImageN {
  using namespace RavlN;

  class FontC {
  public:
    FontC();

    bool IsValid() const;
    // Is this a valid font ?

    Index2dC Center(StringC text) const;
    //: Get the offset to the center of the string.

    Index2dC Size(StringC text) const;
    //: Compute the size of image required to render 'text'.

    UIntT Count() const;
  };


  FontC DefaultFont();
  //: Access default font.

  template<class DataT>
  void DrawText(const FontC &font,const DataT &value,const Index2dC &offset,const StringC &text,ImageC<DataT> &image);

  template<class DataT>
  void DrawTextCenter(const FontC &font,const DataT &value,const Index2dC &offset,const StringC &text,ImageC<DataT> &image);

}

%template(DrawTextByteRGB) RavlImageN::DrawText<RavlImageN::ByteRGBValueC>;
%template(DrawTextByte) RavlImageN::DrawText<RavlN::ByteT>;
%template(DrawTextCenterByteRGB) RavlImageN::DrawTextCenter<RavlImageN::ByteRGBValueC>;
%template(DrawTextCenterByte) RavlImageN::DrawTextCenter<RavlN::ByteT>;
