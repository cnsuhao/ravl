// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Node.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHNODE_HEADER
#define RAVLGUI_OPENSCENEGRAPHNODE_HEADER 1

#include "Ravl/SmartPtr.hh"
#include <osg/ref_ptr>
#include <osg/Node>
#include <osg/NodeCallback>
#include "Ravl/Threads/Signal.hh"

namespace RavlOSGN
{

  //! userlevel=Normal
  //: Node base object.

  class NodeC
  : public RavlN::RCBodyVC
  {
  public:
    NodeC();
    //: Ctor.

    virtual ~NodeC();
    //: Dtor.

    bool EnableCallback(const bool enable = true);
    //: Enabled the callback function.

    RavlN::Signal0C &SigCallback()
    { return m_sigCallback; }
    //: Access the callback signal.
    
    bool BringToFront();
    //: Make sure the node is rendered after all other objects and disable depth testing.

    osg::ref_ptr<osg::Node> Node()
    { return m_node; }
    //: Access the object smart pointer.

    typedef RavlN::SmartPtrC<NodeC> RefT;
    
  protected:
    osg::ref_ptr<osg::Node> m_node;
    osg::ref_ptr<osg::NodeCallback> m_callback;
    RavlN::Signal0C m_sigCallback;
  };

}

#endif
