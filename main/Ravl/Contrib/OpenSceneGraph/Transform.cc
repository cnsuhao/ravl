// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Transform.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Transform.hh"
#include <osg/Transform>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlOSGN
{

  using namespace osg;

  TransformC::TransformC(bool create)
  : GroupC(false)
  {
    if (create)
      m_node = new Transform();
  }

  TransformC::~TransformC()
  {
  }

  TransformPositionAttitudeC::TransformPositionAttitudeC(bool create)
  : TransformC(false)
  {
    if (create)
      m_node = new PositionAttitudeTransform();
  }

  TransformPositionAttitudeC::~TransformPositionAttitudeC()
  {
  }

  bool TransformPositionAttitudeC::SetPosition(const RavlN::Vector3dC &position)
  {
    if (!m_node)
      return false;

    ref_ptr<PositionAttitudeTransform> transformRef = dynamic_cast<PositionAttitudeTransform*>(m_node.get());
    if (!transformRef)
      return false;

    transformRef->setPosition(Vec3(position.X(), position.Y(), position.Z()));

    return true;
  }

  bool TransformPositionAttitudeC::SetAttitude(const RavlN::Quatern3dC &attitude)
  {
    if (!m_node)
      return false;

    ref_ptr<PositionAttitudeTransform> transformRef = dynamic_cast<PositionAttitudeTransform*>(m_node.get());
    if (!transformRef)
      return false;

    transformRef->setAttitude(Quat(attitude[1], attitude[2], attitude[3], attitude[0]));

    return true;
  }

  bool TransformPositionAttitudeC::SetScale(const RavlN::Vector3dC &scale)
  {
    if (!m_node)
      return false;

    ref_ptr<PositionAttitudeTransform> transformRef = dynamic_cast<PositionAttitudeTransform*>(m_node.get());
    if (!transformRef)
      return false;

    transformRef->setScale(Vec3(scale.X(), scale.Y(), scale.Z()));

    return true;
  }


}
