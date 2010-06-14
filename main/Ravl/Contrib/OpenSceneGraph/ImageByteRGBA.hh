// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/ImageByteRGBA.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHIMAGEBYTERGBA_HEADER
#define RAVLGUI_OPENSCENEGRAPHIMAGEBYTERGBA_HEADER 1

#include "Ravl/OpenSceneGraph/Node.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBAValue.hh"
#include "Ravl/RealRange2d.hh"
#include <osg/Geometry>

namespace RavlOSGN
{

  //! userlevel=Normal
  //: Node object containing a ByteRGBAValueC image.

  class ImageByteRGBAC
  : public NodeC
  {
  public:
    ImageByteRGBAC(const RavlN::RealRange2dC &coords = RavlN::RealRange2dC(1, 1));
    //: Ctor.
    //!param: coords - The 2D position of the image in the X/Y plane.

    virtual ~ImageByteRGBAC();
    //: Dtor.

    bool SetImage(RavlImageN::ImageC<RavlImageN::ByteRGBAValueC> &image);
    //: Set the object image.

    bool AlphaImageEnable(bool alphaImageEnable);
    //: Enable transparency for the alpha channel in the image (off by default for performance reasons).

    bool SetAlpha(float alpha);
    //: Set an alpha value for the entire image.

    typedef RavlN::SmartPtrC<ImageByteRGBAC> RefT;

  protected:
    bool UpdateAlphaRenderState(osg::ref_ptr<osg::StateSet> &stateSetRef);

    osg::ref_ptr<osg::Geometry> m_geometry;
    RavlImageN::ImageC<RavlImageN::ByteRGBAValueC> m_image;
    bool m_alphaImageEnabled;
    float m_alpha;
  };

}

#endif
