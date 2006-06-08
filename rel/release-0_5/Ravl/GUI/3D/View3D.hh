// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLGUI3D_VIEW3D_HEADER
#define RAVLGUI3D_VIEW3D_HEADER 1
///////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/GUI/3D/View3D.hh"
//! lib=RavlGUI3D
//! userlevel=Normal
//! docentry="Ravl.GUI.3D"
//! author="James Smith"

#include "Ravl/GUI/Canvas3D.hh"
#include "Ravl/GUI/Table.hh"
#include "Ravl/GUI/Menu.hh"
#include "Ravl/GUI/MenuCheck.hh"

namespace RavlGUIN {

  class MouseEventC;
  class View3DC;
  
  //: 3D Viewer widget.
  
  class View3DBodyC
    : public TableBodyC
  {
  public:
    View3DBodyC(int sx,int sy);
    //: Default constructor.
    
    bool Put(const DObject3DC &r);
    //: Put render instructon into pipe.
    
    DObjectSet3DC &Scene()
      { return scene; }
    //: Access current scene.
    
    void DoSetup();
    //: Fit and centre output if appropriate
    
    void DoFit(bool& bRefresh);
    //: Fit object to view
    
    void DoCenter(bool& bRefresh);
    //: Center output.
    
    void AutoFit(bool &val);
    //: Auto fit output.
    
    void AutoCenter(bool &val);
    //: Auto center output.
    
  protected:
    void MousePress(MouseEventC &me);
    //: Handle button press.
    
    void MouseRelease(MouseEventC &me);
    //: Handle button release.
    
    void MouseMove(MouseEventC &me);
    //: Handle mouse move.
    
    bool Create();
    //: Setup widget.
    
    void Refresh();
    //: Refresh display.
    
    void ResetCamera();
    //: Resets the camera position.
    
    void SetRenderMode(int& iOption);
    //: Sets the rendering mode
    // Reads value from the appropriate render mode menu item, and updates the other menu options appropriately.
    
    void FrontFaces(bool& bFront) {m_bFront = bFront; SetCullMode();}
    //: Enable or disable frontfaces
    
    void BackFaces(bool& bBack) {m_bBack = bBack; SetCullMode();}
    //: Enable or disable backfaces
    
    void SetCullMode(void);
    //: Sets the face culling mode based on member variables
    
    void InitGL(void);
    //: Sets up GL context
    
    void NewFrame(void);
    //: Sets up for a new frame
    
    void SetCamera(void);
    //: Rotates the camera
    
    bool sceneComplete;
    DObjectSet3DC scene; // List of current render instructions.
    Canvas3DC canvas;
    MenuC backMenu;
    Vector3dC viewObject; // looking at point.
    Vector3dC viewPoint;  // Where we are.
    bool useRotate;
    Vector3dC viewRotate;    // Rotation to apply.
    RealT fov;
    
    bool initDone; // Has initalization been completed ?
    
    // Display settings
    bool m_bAutoCenter;
    bool m_bAutoFit;
    
    // Mouse position storage
    Index2dC m_pixButton0Pos;
    Index2dC m_pixButton1Pos;
    
    // Camera position params
    RealT m_fXRotation;
    RealT m_fYRotation;
    RealT m_fXTranslation;
    RealT m_fYTranslation;
    RealT m_fZoom;
    
    // Render mode menu option handles
    MenuCheckItemC m_oRenderOpts[4];
    
    // Culling options
    bool m_bFront;
    bool m_bBack;
    
    friend class View3DC;
  };
  

  //: 3D Viewer widget.
  
  class View3DC
    : public TableC
  {
  public:
    View3DC()
      {}
    //: Default constructor.
    // creates an invalid handle.
    
    View3DC(int sx,int sy)
      : TableC(*new View3DBodyC(sx,sy))
      {}
    //: Constructor.
    
  protected:
    View3DC(View3DBodyC &bod)
      : TableC(bod)
      {}
    //: Body constructor.
    
    View3DBodyC &Body() 
      { return static_cast<View3DBodyC &>(WidgetC::Body()); }
    //: Access body.
  
    const View3DBodyC &Body() const
      { return static_cast<const View3DBodyC &>(WidgetC::Body()); }
    //: Access body.
    
    void InitGL(void)
      { Body().InitGL(); }
    //: Sets up GL context
    
    void NewFrame(void)
      { Body().NewFrame(); }
    //: Sets up for a new frame
    
    void SetCamera(void)
      { Body().SetCamera(); }
    //: Rotates the camera
    
  public:
    
    DObjectSet3DC &Scene()
      { return Body().Scene(); }
    //: Access current scene.
    
    void DoFit(bool& bRefresh)
      { Body().DoFit(bRefresh); }
    //: Auto fit output.
    
    void DoCenter(bool& bRefresh)
      { Body().DoCenter(bRefresh); }
    //: Auto center output.
    
    void AutoFit(bool &val)
      { Body().AutoFit(val); }
    //: Auto fit output.
    
    void AutoCenter(bool &val)
      { Body().AutoCenter(val); }
    //: Auto center output.  
    
    friend class View3DBodyC;
  };
  
}



#endif