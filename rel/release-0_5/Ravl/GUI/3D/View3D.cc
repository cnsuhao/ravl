// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlGUI3D

#include "Ravl/GUI/View3D.hh"
#include "Ravl/GUI/Util.hh"
#include "Ravl/GUI/Menu.hh"
#include "Ravl/GUI/MenuCheck.hh"
#include "Ravl/GUI/MouseEvent.hh"
#include "Ravl/StdMath.hh"
#include "Ravl/StdConst.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlGUIN {

  //: Default constructor.
  
  View3DBodyC::View3DBodyC(int sx,int sy)
    : TableBodyC(1,1),
      canvas(sx,sy),
      viewObject(0,0,0),
      viewPoint(0,0,3),
      useRotate(true),
      viewRotate(0,0,0),
      fov(90),
      initDone(false),
      m_bAutoCenter(false),
      m_bAutoFit(false),
      m_pixButton0Pos(0,0),
      m_pixButton1Pos(0,0),
      m_fXRotation(0),
      m_fYRotation(0),
      m_fXTranslation(0),
      m_fYTranslation(0),
      m_fZoom(0),
      m_bFront(true),
      m_bBack(false)
  {
    ONDEBUG(cerr << "View3DBodyC::View3DBodyC(), Called. \n");
  }
  
  //: Fit and centre output if appropriate
  void View3DBodyC::DoSetup() {
    bool bFalse = false;
    if(m_bAutoCenter) DoCenter(bFalse);
    if(m_bAutoFit)    DoFit(bFalse);
  }
  
  //: Fit object to view
  void View3DBodyC::DoFit(bool& bRefresh) {
    ONDEBUG(cerr << "View3DBodyC::DoFit(), Called. \n");
    RealT dist = viewObject.EuclidDistance(viewPoint);
    RealT extent = scene.Extent() * 1.3;
    if(dist <= 0)
      dist = 0.01;
    fov = ATan(extent/dist) * (180 / RavlConstN::pi);
    ONDEBUG(cerr << "Set fov to " << fov << "\n");
    canvas.Put(DViewPoint3DC(fov,viewPoint,viewObject));
    if (bRefresh) Refresh();
  }
  
  //: Center output.
  void View3DBodyC::DoCenter(bool& bRefresh) {
    ONDEBUG(cerr << "View3DBodyC::DoCenter(), Called. \n");
    viewObject = scene.Center();
    canvas.Put(DViewPoint3DC(fov,viewPoint,viewObject));
    ResetCamera();
    if (bRefresh) Refresh();
  }
  
  //: Auto fit output.
  void View3DBodyC::AutoFit(bool &val) {
    m_bAutoFit = val;
    bool bTrue = true;
    if(m_bAutoFit) DoFit(bTrue);
  }
  
  //: Auto center output.
  void View3DBodyC::AutoCenter(bool &val) {
    m_bAutoCenter = val;
    bool bTrue = true;
    if(m_bAutoCenter) DoCenter(bTrue);
  }
  
  //: Handle button press.
  void View3DBodyC::MousePress(MouseEventC &me) {
    ONDEBUG(cerr << "View3DBodyC::MousePress(), Called. '" << me.HasChanged(0) << " " << me.HasChanged(1) << " " << me.HasChanged(2) <<"' \n");
    if(me.HasChanged(0)) 
      m_pixButton0Pos = me.Position();
    else if(me.HasChanged(1)) 
      m_pixButton1Pos = me.Position();
    else if(me.HasChanged(2)) {
    ONDEBUG(cerr << "Show menu. \n");
    backMenu.Popup(); 
    }
  }
  
  //: Handle button release.
  void View3DBodyC::MouseRelease(MouseEventC &me) {
    ONDEBUG(cerr << "View3DBodyC::MouseRelease(), Called. '" << me.HasChanged(0) << " " << me.HasChanged(1) << " " << me.HasChanged(2) <<"' \n");
  }
  
  //: Handle mouse move.
  void View3DBodyC::MouseMove(MouseEventC &me) {
    //ONDEBUG(cerr << "View3DBodyC::MouseMove(), Called. '" << me.HasChanged(0) << " " << me.HasChanged(1) << " " << me.HasChanged(2) <<"' \n");
    //ONDEBUG(cerr << "View3DBodyC::MouseMove(),         '" << me.IsPressed(0) << " " << me.IsPressed(1) << " " << me.IsPressed(2) <<"' \n");
    //cerr << "View3DBodyC::MouseMove(), Called. \n";
    
  // X rotation limit values
    const RealT upper_limit = 90;
    const RealT lower_limit = -90;
    
    // Zoom in and out if buttons 0 and 1 are pressed
    if (me.IsPressed(0) && me.IsPressed(1)) {
      // Calculate change
      Index2dC change = me.Position() -  m_pixButton0Pos; 
      
      // Calculate individual rotations
      m_fZoom += change.Col();
      
      // Store new position
      m_pixButton0Pos = m_pixButton1Pos = me.Position();
      
      // Update display
      Refresh();
    }
    
    // Rotate when button 0 pressed
    else if(me.IsPressed(0)) {
      // Calculate change
      Index2dC change = me.Position() -  m_pixButton0Pos; 
      
      // Calculate individual rotations
      m_fXRotation += change.Col();
      m_fYRotation += change.Row();
      
      // Limit X rotation
      if (m_fXRotation > upper_limit) m_fXRotation = upper_limit;
      else if (m_fXRotation < lower_limit) m_fXRotation = lower_limit;
      
      // Store new position
      m_pixButton0Pos = me.Position();
      
      // Update display
      Refresh();
    }
    
    // Translate when button 1 pressed
    else if (me.IsPressed(1)) {
      // Calculate change
      Index2dC change = me.Position() -  m_pixButton1Pos; 
      
      // Calculate individual translations (Y is inverted)
      m_fXTranslation += (RealT)change.Row() / 100.0;
      m_fYTranslation -= (RealT)change.Col() / 100.0;
      
      // Store new position
      m_pixButton1Pos = me.Position();
      
      // Update display
      Refresh();
    }
  }
  
  //: Setup widget.
  bool View3DBodyC::Create() {
    ONDEBUG(cerr << "View3DBodyC::Create(), Called. \n");
    
    
    ConnectRef(canvas.Signal("button_press_event"),*this,&View3DBodyC::MousePress);
    ConnectRef(canvas.Signal("button_release_event"),*this,&View3DBodyC::MouseRelease);
    ConnectRef(canvas.Signal("motion_notify_event"),*this,&View3DBodyC::MouseMove);
    ConnectRef(canvas.Signal("expose_event"),*this,&View3DBodyC::Refresh);
    
    if(!canvas.Create()) {
      // Get this sorted out early.
      cerr << "View3DBodyC::Create(), ERROR: 3DCanvas create failed. \n";
      return false;
    }
    
    ONDEBUG(cerr << "View3DBodyC::Create(), Setting up canvas intialization. \n");
    
    // Setup render options
    m_oRenderOpts[0] = MenuCheckItemC("Points",false);
    m_oRenderOpts[1] = MenuCheckItemC("Wire",false);
    m_oRenderOpts[2] = MenuCheckItemC("Flat",false);
    m_oRenderOpts[3] = MenuCheckItemC("Smooth",true);
    for (int i=0; i<4; i++) {
      ConnectRef(m_oRenderOpts[i].SigSelected(),*this,&View3DBodyC::SetRenderMode,i);
    }
    
    // Setup backmenu.
    bool bTrue = true;
    bool bTextureStatus = true;
    bool bLightingStatus = true;
    
    MenuC renderMenu("Render",
		     m_oRenderOpts[0] +
		     m_oRenderOpts[1] +
		     m_oRenderOpts[2] +
		     m_oRenderOpts[3] +
		     MenuItemSeparator() +
		     MenuCheckItemR("Texturing",bTextureStatus,canvas,&Canvas3DC::SetTextureMode) +
		     MenuCheckItemR("Lighting",bLightingStatus,canvas,&Canvas3DC::SetLightingMode)
		     );
    
    MenuC facesMenu("Faces",
		    MenuCheckItemR("Front",m_bFront,*this,&View3DBodyC::FrontFaces) +
		    MenuCheckItemR("Back",m_bBack,*this,&View3DBodyC::BackFaces)
		    );  
    
    backMenu = MenuC("back",
		     MenuItemRef("Center",*this,&View3DBodyC::DoCenter,bTrue) +
		     MenuItemRef("Fit",*this,&View3DBodyC::DoFit,bTrue) +
		     MenuCheckItemR("Auto Center",*this,&View3DBodyC::AutoCenter) +
		     MenuCheckItemR("Auto Fit",*this,&View3DBodyC::AutoFit) +
		     MenuItemSeparator() +
		     renderMenu +
		     facesMenu
		     );
    
    
    TableBodyC::AddObject(canvas,0,1,0,1);
    
    if(!TableBodyC::Create()) {
      cerr << "WARNING: ViewGeometry2DBodyC::Create(), failed. \n";
      return false;
    }
    
    ONDEBUG(cerr << "View3DBodyC::Create(), Doing setup. \n");
    
    // Initialise OpenGL
    canvas.Put(DOpenGLC(SignalEventMethod0C<View3DC>(View3DC(*this),&View3DC::InitGL)));
    canvas.SetTextureMode(bTextureStatus);
    canvas.SetLightingMode(bLightingStatus);
    
    // Setup lights and cameras
    canvas.Light(RealRGBValueC(1,1,1),Point3dC(0,0,10));
    canvas.ViewPoint(90,viewPoint); // Setup view point.
    
    ONDEBUG(cerr << "View3DBodyC::Create(), Done. \n");
    return true;
  }
  
  //: Put render instructon into pipe.
  
  bool View3DBodyC::Put(const DObject3DC &r) {
    ONDEBUG(cerr << "View3DBodyC::Put(), Called. \n");
    if(sceneComplete) {
      scene = DObjectSet3DC(true);
      sceneComplete = false;
    }
    DSwapBuff3DC swapBuff(r);
    if(swapBuff.IsValid())
      sceneComplete = true;
    else {
      if(!scene.IsValid()) 
	scene = DObjectSet3DC(true);    
      if(r.IsValid())
	scene += r;
    }
    if(m_bAutoFit || m_bAutoCenter) DoSetup();
    Refresh();
    ONDEBUG(cerr << "View3DBodyC::Put(), Done. \n");
    return true;
  }
  
  //: Put End Of Stream marker.
  void View3DBodyC::PutEOS() {
    sceneComplete = true;
  }
  
  //: Is port ready for data ?
  bool View3DBodyC::IsPutReady() const {
    return true;
  }
  
  //: Refresh display.
  void View3DBodyC::Refresh() {
    ONDEBUG(cerr << "View3DBodyC::Refresh(), Called. " << ((void *) widget) << "\n");
    if(!initDone)
      return ; // Can't do anything before the setup is complete.
    
    canvas.Put(DOpenGLC(SignalEventMethod0C<View3DC>(View3DC(*this),&View3DC::NewFrame)));
    
    canvas.Put(DOpenGLC(SignalEventMethod0C<View3DC>(View3DC(*this),&View3DC::SetCamera)));
    
    // Render scene
    if(scene.IsValid())
      canvas.Put(scene);
    canvas.SwapBuffers();  
  }
  
  //: Reset the camera position
  void View3DBodyC::ResetCamera() {
    m_fXRotation = 0;
    m_fYRotation = 0;
    m_fXTranslation = 0;
    m_fYTranslation = 0;
    m_fZoom = 0;
    return;
  }
  
  void View3DBodyC::SetRenderMode(int& iOption) {
    bool bVal = m_oRenderOpts[iOption].IsActive();
    if (bVal) {
      for (int i=0; i<4; i++) {
	if (i!=iOption) {
	  m_oRenderOpts[i].SetActive(false);
	}
	switch (iOption) {
	case 0:
	  canvas.SetRenderMode(C3D_POINT);
	  break;
	case 1:
	  canvas.SetRenderMode(C3D_WIRE);
	  break;
	case 2:
	  canvas.SetRenderMode(C3D_FLAT);
	  break;
	case 3:
	  canvas.SetRenderMode(C3D_SMOOTH);
	  break;
	default:
	  break;
	}
      }
    }
    else {
      int iNumTrue = 0;
      for (int i=0; i<4; i++) {
	if (i!=iOption && m_oRenderOpts[i].IsActive()) iNumTrue++;
      }
      if (iNumTrue == 0) {
	m_oRenderOpts[iOption].SetActive(true);
      }
    }
  }
  
  void View3DBodyC::InitGL() {
    ONDEBUG(cerr << "View3DBodyC::InitGL(), Called. \n");
    // Set up culling
    SetCullMode();
    // Enable depth testing
    glEnable(GL_DEPTH_TEST);
    // Init shade model
    glShadeModel(GL_SMOOTH);
    canvas.SetRenderMode(C3D_SMOOTH);
    // Let everyone know we're ready to go.
    initDone = true;
    return;
  }
  
  void View3DBodyC::SetCullMode() {
    ONDEBUG(cerr << "View3DBodyC::SetCullMode(), Called. \n");
    
    if (m_bFront) {
      if (m_bBack) {
	glDisable(GL_CULL_FACE);
      }
      else {
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
      }
    }
    else {
      if (m_bBack) {
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);      
      }
      else {
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT_AND_BACK);
      }    
    }
  }
  
  void View3DBodyC::NewFrame() {
    // Clear buffers
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    // Reset matrix
    glLoadIdentity();
    // Done
    return;
  }
  
  void View3DBodyC::SetCamera() {
    // Rotate scene
    glRotated(m_fXRotation,1,0,0);
    glRotated(m_fYRotation,0,1,0);
    
    // Translate scene
    //Vector3dC vecTranslation(m_fXTranslation,0,0);
    //vecTranslation = vecTranslation.Rotation(Vector3dC(0,1,0),m_fYRotation);
    //glTranslated(vecTranslation.X(),m_fYTranslation,vecTranslation.Z());
    //glTranslated(m_fXTranslation,m_fYTranslation,0);  
    
  // Done
    return;
  }

}