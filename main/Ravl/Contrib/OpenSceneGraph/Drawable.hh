// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Drawable.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHDRAWABLE_HEADER
#define RAVLGUI_OPENSCENEGRAPHDRAWABLE_HEADER 1

#include "Ravl/SmartPtr.hh"
#include <osg/ref_ptr>
#include <osg/Drawable>

namespace RavlOSGN
{

  //! userlevel=Normal
  //: Drawable object base class.

  class DrawableC
  : public RavlN::RCBodyVC
  {
  public:
    DrawableC();
    //: Ctor.

    virtual ~DrawableC();
    //: Dtor.

    osg::ref_ptr<osg::Drawable> Drawable()
    { return m_drawable; }
    //: Access the object smart pointer.
    
    typedef RavlN::SmartPtrC<DrawableC> RefT;

  protected:
    osg::ref_ptr<osg::Drawable> m_drawable;
  };

}

#endif
