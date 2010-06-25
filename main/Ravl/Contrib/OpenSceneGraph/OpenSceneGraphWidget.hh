// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/OpenSceneGraphWidget.hh"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"
//! docentry = "Ravl.API.Graphics.OpenSceneGraph"
//! example = "exOpenSceneGraphWidget.cc"

#ifndef RAVLGUI_OPENSCENEGRAPHWIDGET_HEADER
#define RAVLGUI_OPENSCENEGRAPHWIDGET_HEADER 1

#include "Ravl/GUI/Widget.hh"
#include "Ravl/OpenSceneGraph/Node.hh"
#include <osgGtk/ViewerGtk.h>
#include <osgGtk/GraphicsWindowGtk.h>
#include "Ravl/Image/RealRGBAValue.hh"
#include "Ravl/XMLFactory.hh"

namespace RavlOSGN
{

  //! userlevel=Normal
  //: GTK widget containing an OpenSceneGraph viewer.

  class OpenSceneGraphWidgetBodyC
  : public RavlGUIN::WidgetBodyC
  {
  public:
    OpenSceneGraphWidgetBodyC(int width, int height);
    //: Ctor.
    //!param: width - Initial width of the widget.
    //!param: height - Initial height of the widget.

    OpenSceneGraphWidgetBodyC(const RavlN::XMLFactoryContextC &factory);
    //: Ctor.
    //!param: factory - Construct from a factory context.

    virtual ~OpenSceneGraphWidgetBodyC();
    //: Dtor.

    virtual bool Create()
    { return CommonCreate(); }

    virtual bool Create(GtkWidget *newWidget)
    { return CommonCreate(newWidget); }

    bool SetScene(const NodeC::RefT &node);
    //: Set the scene graph root object.

    bool SetColour(const RavlImageN::RealRGBAValueC &colour);
    //: Set the viewer background colour.

  protected:
    bool CommonCreate(GtkWidget *newWidget = NULL);

    bool OnConfigure();
    //: Change the viewport on resize.

    bool OnDestroy();
    //: Stop running on destory.

    int m_width, m_height;
    osg::ref_ptr<osgViewer::ViewerGtk> m_osgViewer;
    osg::ref_ptr<osgViewer::GraphicsWindowGtk> m_osgWindow;

    NodeC::RefT m_sceneNode;
    RavlImageN::RealRGBAValueC m_clearColour;
  };

  //! userlevel=Normal
  //: OpenSceneGraph widget.
  
  class OpenSceneGraphWidgetC
  : public RavlGUIN::WidgetC
  {
  public:
    OpenSceneGraphWidgetC()
    {}
    //: Default ctor.
    // Creates an invalid handle.

    OpenSceneGraphWidgetC(const RavlN::XMLFactoryContextC &factory)
    : WidgetC(*new OpenSceneGraphWidgetBodyC(factory))
    {}
    //: Ctor.
    //!param: factory - Construct from a factory context.

    OpenSceneGraphWidgetC(int width, int height)
    : WidgetC(*new OpenSceneGraphWidgetBodyC(width, height))
    {}
    //: Ctor.
    //!param: width - Initial width of the widget.
    //!param: height - Initial height of the widget.

    OpenSceneGraphWidgetC(OpenSceneGraphWidgetBodyC &body)
    : WidgetC(body)
    {}
    //: Body ctor.

    bool SetScene(const NodeC::RefT &node)
    { return Body().SetScene(node); }
    //: Set the scene graph root object.

    bool SetColour(const RavlImageN::RealRGBAValueC &colour)
    { return Body().SetColour(colour); }
    //: Set the viewer background colour.
    
  protected:
    OpenSceneGraphWidgetBodyC &Body()
    { return static_cast<OpenSceneGraphWidgetBodyC &>(RavlGUIN::WidgetC::Body()); }

    const OpenSceneGraphWidgetBodyC &Body() const
    { return static_cast<const OpenSceneGraphWidgetBodyC &>(RavlGUIN::WidgetC::Body()); }
  };

}

#endif
