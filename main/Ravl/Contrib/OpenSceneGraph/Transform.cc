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
#include "Ravl/OpenSceneGraph/TypeConvert.hh"
#include "Ravl/XMLFactoryRegister.hh"

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

  //: XML constructor.
  TransformC::TransformC(const XMLFactoryContextC &context)
  {
    m_node = new Transform();
    GroupC::Setup(context);
  }

  TransformC::~TransformC()
  {
  }
  
  //: Do setup from factory
  bool TransformC::Setup(const XMLFactoryContextC &factory)
  {
    GroupC::Setup(factory);
    return true;
  }

  //: Called when owner handles drop to zero.
  void TransformC::ZeroOwners()
  {
    GroupC::ZeroOwners();
  }
  
  // ------------------------------------------------------------------------
  
  TransformPositionAttitudeC::TransformPositionAttitudeC(const XMLFactoryContextC &context)
  : TransformC(false)
  {
    m_node = new PositionAttitudeTransform();
    Setup(context);
  }
    //: XML constructor.


  TransformPositionAttitudeC::TransformPositionAttitudeC(bool create)
  : TransformC(false)
  {
    if (create)
      m_node = new PositionAttitudeTransform();
  }

  TransformPositionAttitudeC::~TransformPositionAttitudeC()
  {
  }

  //: Do setup from factory
  
  bool TransformPositionAttitudeC::Setup(const XMLFactoryContextC &factory)
  {
    TransformC::Setup(factory);
    
    RavlN::Vector3dC position(0,0,0);
    if(factory.Attribute("position",position)) {
      SetPosition(position);
    }
    
    RavlN::Vector3dC scale(1.0,1.0,1.0);
    if(factory.Attribute("scale",scale)) {
      SetScale(scale);
    }
    
    return true;
  }
  
  bool TransformPositionAttitudeC::SetPosition(const RavlN::Vector3dC &position)
  {
    if (!m_node)
      return false;

    ref_ptr<PositionAttitudeTransform> transformRef = dynamic_cast<PositionAttitudeTransform*>(m_node.get());
    if (!transformRef)
      return false;

    transformRef->setPosition(MakeVec3(position));

    return true;
  }

  bool TransformPositionAttitudeC::GetPosition(RavlN::Vector3dC &position)
  {
    if (!m_node)
      return false;

    ref_ptr<PositionAttitudeTransform> transformRef = dynamic_cast<PositionAttitudeTransform*>(m_node.get());
    if (!transformRef)
      return false;

    Vec3 vecPosition = transformRef->getPosition();
    position = MakeVector3d(vecPosition);
    
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
  
  //: Called when owner handles drop to zero.
  void TransformPositionAttitudeC::ZeroOwners()
  {
    TransformC::ZeroOwners();
  }

  void LinkTransform()
  {}
  
  static RavlN::XMLFactoryRegisterConvertC<TransformPositionAttitudeC,TransformC> g_registerXMLFactoryTransformPositionAttitude("RavlOSGN::TransformPositionAttitudeC");
  static RavlN::XMLFactoryRegisterConvertC<TransformC,GroupC> g_registerXMLFactoryTransform("RavlOSGN::TransformC");

}
