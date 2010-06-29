// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Box.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHBOX_HEADER
#define RAVLGUI_OPENSCENEGRAPHBOX_HEADER 1

#include "Ravl/OpenSceneGraph/Drawable.hh"
#include "Ravl/Vector3d.hh"
#include "Ravl/Image/RealRGBAValue.hh"
#include <osg/Shape>

namespace RavlOSGN
{

  //! userlevel=Normal
  //: Box object.

  class BoxC
  : public DrawableC
  {
  public:
    BoxC(const RavlN::Vector3dC &position = RavlN::Vector3dC(0, 0, 0),
            const RavlN::Vector3dC &size = RavlN::Vector3dC(1,1,1));
    //: Ctor.

    virtual ~BoxC();
    //: Dtor.

    bool SetColour(const RavlImageN::RealRGBAValueC &colour);
    //: Set the Box colour.

    bool SetPosition(const RavlN::Vector3dC &position);
    //: Set the Box centre position.

    bool SetSize(const RavlN::Vector3dC &size);
    //: Set the Box radius.

    typedef RavlN::SmartPtrC<BoxC> RefT;
    
  protected:
    osg::ref_ptr<osg::Box> m_box;
  };

}

#endif
