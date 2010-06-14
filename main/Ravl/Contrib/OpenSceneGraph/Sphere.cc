// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Sphere.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Sphere.hh"
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

  SphereC::SphereC(const RavlN::Vector3dC &position, RavlN::RealT radius)
  {
    m_sphere = new Sphere(Vec3(position.X(), position.Y(), position.Z()), radius);
    m_drawable = new ShapeDrawable(m_sphere);
  }

  SphereC::~SphereC()
  {
  }

  bool SphereC::SetColour(const RavlImageN::RealRGBAValueC &colour)
  {
    if (!m_drawable)
      return false;

    ref_ptr<ShapeDrawable> shapeDrawableRef = dynamic_cast<ShapeDrawable*>(m_drawable.get());
    if (!shapeDrawableRef)
      return false;

    shapeDrawableRef->setColor(Vec4(colour.Red(), colour.Green(), colour.Blue(), colour.Alpha()));

    return true;
  }

  bool SphereC::SetPosition(const RavlN::Vector3dC &position)
  {
    if (!m_sphere)
      return false;

    m_sphere->setCenter(Vec3(position.X(), position.Y(), position.Z()));

    return true;
  }

  bool SphereC::SetSize(RavlN::RealT radius)
  {
    if (!m_sphere)
      return false;

    m_sphere->setRadius(radius);

    return false;
  }

}
