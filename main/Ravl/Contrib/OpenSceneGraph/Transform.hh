// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Transform.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHTRANSFORM_HEADER
#define RAVLGUI_OPENSCENEGRAPHTRANSFORM_HEADER 1

#include "Ravl/OpenSceneGraph/Group.hh"
#include "Ravl/Vector3d.hh"
#include "Ravl/Quatern3d.hh"
#include <osg/Transform>
#include <osg/PositionAttitudeTransform>

namespace RavlOSGN
{

  //! userlevel=Normal
  //: Transform object.

  class TransformC
  : public GroupC
  {
  public:
    TransformC(bool create = true);
    //: Ctor.
    //!param: create - If true, a new transform object will be allocated.

    virtual ~TransformC();
    //: Dtor.

    typedef RavlN::SmartPtrC<TransformC> RefT;

  protected:
  };

  //! userlevel=Normal
  //: Position and attitude transform object.

  class TransformPositionAttitudeC
  : public TransformC
  {
  public:
    TransformPositionAttitudeC(bool create = true);
    //!param: create - If true, a new position and attitude transform object will be allocated.

    virtual ~TransformPositionAttitudeC();
    //: Dtor.

    bool SetPosition(const RavlN::Vector3dC &position);
    //: Set the transform position.

    bool GetPosition(RavlN::Vector3dC &position);
    //: Get the transform position.

    bool SetAttitude(const RavlN::Quatern3dC &attitude);
    //: Set the transform attitude.

    bool SetScale(const RavlN::Vector3dC &scale);
    //: Set the transform scale.

    typedef RavlN::SmartPtrC<TransformPositionAttitudeC> RefT;

  protected:
  };

}

#endif
