// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Box.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Box.hh"
#include <osg/Shape>
#include <osg/ShapeDrawable>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlOSGN
{

  using namespace osg;

  BoxC::BoxC(const RavlN::Vector3dC &position, const RavlN::Vector3dC &size)
  {
    m_box = new osg::Box(Vec3(position.X(), position.Y(), position.Z()),size.X(), size.Y(), size.Z());
    m_drawable = new ShapeDrawable(m_box);
  }

  BoxC::~BoxC()
  {
  }

  bool BoxC::SetColour(const RavlImageN::RealRGBAValueC &colour)
  {
    if (!m_drawable)
      return false;

    ref_ptr<ShapeDrawable> shapeDrawableRef = dynamic_cast<ShapeDrawable*>(m_drawable.get());
    if (!shapeDrawableRef)
      return false;

    shapeDrawableRef->setColor(Vec4(colour.Red(), colour.Green(), colour.Blue(), colour.Alpha()));

    return true;
  }

  bool BoxC::SetPosition(const RavlN::Vector3dC &position)
  {
    if (!m_box)
      return false;

    m_box->setCenter(Vec3(position.X(), position.Y(), position.Z()));

    return true;
  }

  bool BoxC::SetSize(const RavlN::Vector3dC &size)
  {
    if (!m_box)
      return false;

    m_box->setHalfLengths(Vec3(size.X(), size.Y(), size.Z()));

    return true;
  }

}
