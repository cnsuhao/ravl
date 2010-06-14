// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Node.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Node.hh"
#include <osg/Node>
#include <osg/StateSet>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlOSGN
{

  using namespace osg;

  NodeC::NodeC()
  {
  }

  NodeC::~NodeC()
  {
  }
  
  bool NodeC::BringToFront()
  {
    if (!m_node)
      return false;

    ref_ptr<StateSet> stateSetRef = m_node->getOrCreateStateSet();

    stateSetRef->setMode(GL_DEPTH_TEST, StateAttribute::OFF);
    stateSetRef->setRenderBinDetails(10, "RenderBin");

    m_node->setStateSet(stateSetRef);

    return true;
  }

}
