// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/exOpenSceneGraphWidget.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/GUI/Manager.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/Option.hh"
#include "Ravl/GUI/Window.hh"
#include "Ravl/GUI/Button.hh"
#include "Ravl/GUI/LBox.hh"
#include "Ravl/DP/FileFormatIO.hh"
#include "Ravl/GUI/MouseEvent.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OpenSceneGraph/OpenSceneGraphWidget.hh"
#include "Ravl/Image/RealRGBAValue.hh"
#include "Ravl/OpenSceneGraph/Geode.hh"
#include "Ravl/OpenSceneGraph/Sphere.hh"
#include "Ravl/OpenSceneGraph/Transform.hh"
#include "Ravl/OpenSceneGraph/ImageByteRGBA.hh"
#include "Ravl/OpenSceneGraph/ModelFile.hh"
#include "Ravl/OpenSceneGraph/Text.hh"
#include "Ravl/OpenSceneGraph/HUD.hh"

using namespace RavlN;
using namespace RavlGUIN;
using namespace RavlOSGN;

bool pressFunc(MouseEventC &me) {
  cerr << "Press " << me.Row() << " " << me.Col() << "\n";
  return true;
}

bool releaseFunc(MouseEventC &me) {
  cerr << "Release " << me.Row() << " " << me.Col() << "\n";
  return true;
}
bool CBSetImage(ImageByteRGBAC::RefT &imageRef)
{
  ImageC<ByteRGBAValueC> image;
  if (Load(PROJECT_OUT "/share/RAVL/pixmaps/CalibrationChart.png", image, "", true))
    imageRef->SetImage(image);
  imageRef->SetAlpha(0.5);
//  imageRef->AlphaImageEnable(true);
  return true;
}

int DoMain(int argc, char *argv[])
{
  Manager.Init(argc, argv);

  OptionC opts(argc, argv);
  opts.Check();

  WindowC win(100, 100, "OpenSceneGraph");
  OpenSceneGraphWidgetC osgWidget(100, 100);
  Connect(osgWidget.Signal("button_press_event"), &pressFunc);
  Connect(osgWidget.Signal("button_release_event"), &releaseFunc);

  RavlGUIN::LBoxC vbox = VBox(osgWidget);

  // Root object
  GroupC::RefT groupRef = new GroupC();

  // Add a sphere
  TransformPositionAttitudeC::RefT transformSphereRef = new TransformPositionAttitudeC();
  groupRef->AddChild(transformSphereRef.BodyPtr());

  SphereC::RefT sphereRef = new SphereC();
  sphereRef->SetColour(RavlImageN::RealRGBAValueC(0.5, 0.0, 0.0, 1.0));

  GeodeC::RefT geodeRef = new GeodeC();
  geodeRef->AddDrawable(sphereRef.BodyPtr());

  transformSphereRef->AddChild(geodeRef.BodyPtr());

  // Add an image
  TransformPositionAttitudeC::RefT transformImageRef = new TransformPositionAttitudeC();
  transformImageRef->SetPosition(RavlN::Vector3dC(1, 0, 0));
  transformImageRef->SetAttitude(RavlN::Quatern3dC(RavlN::Vector3dC(1, 0, 0), RavlConstN::pi / 2.0));
  groupRef->AddChild(transformImageRef.BodyPtr());

  ImageByteRGBAC::RefT imageRef = new ImageByteRGBAC(RealRange2dC(1, 1));

  ImageC<ByteRGBAValueC> image;
  if (Load(PROJECT_OUT "/share/RAVL/pixmaps/monkey.ppm", image, "", true))
    imageRef->SetImage(image);
  transformImageRef->AddChild(imageRef.BodyPtr());

  vbox.Add(Button("Change Image", &CBSetImage, imageRef));

  // Add some text
  TransformPositionAttitudeC::RefT transformTextRef = new TransformPositionAttitudeC();
  transformTextRef->SetPosition(RavlN::Vector3dC(1.5, 0, 1));
  groupRef->AddChild(transformTextRef.BodyPtr());

  TextC::RefT textRef = new TextC("Some Text");
  textRef->SetSize(0.5);
  textRef->SetAlignment(TextAlignCentreBaseLine);
  textRef->SetAxisAlignment(TextAxisScreen);
  textRef->SetColour(RavlImageN::RealRGBAValueC(0.5, 0.5, 0.5, 1.0));

  GeodeC::RefT textGeodeRef = new GeodeC();
//  textGeodeRef->BringToFront();
  textGeodeRef->AddDrawable(textRef.BodyPtr());

  transformTextRef->AddChild(textGeodeRef.BodyPtr());

  HUDC::RefT hudRef = new HUDC(RealRange2dC(0, 768, 0, 1024));
  groupRef->AddChild(hudRef.BodyPtr());

  TextC::RefT textHUDRef = new TextC("HUUUD");
//  textHUDRef->SetFont("/usr/share/fonts/truetype/DejaVuSans.ttf");
  textHUDRef->SetSize(100);
  textHUDRef->SetPosition(Vector3dC(100, 100, -1));
  textHUDRef->SetAlignment(TextAlignCentreBaseLine);
  textHUDRef->SetAxisAlignment(TextAxisScreen);
  textHUDRef->SetColour(RavlImageN::RealRGBAValueC(0.0, 0.0, 0.0, 1.0));

  GeodeC::RefT textHUDGeodeRef = new GeodeC();
  textHUDGeodeRef->AddDrawable(textHUDRef.BodyPtr());
  textHUDGeodeRef->BringToFront();
  hudRef->AddChild(textHUDGeodeRef.BodyPtr());

  osgWidget.SetScene(groupRef.BodyPtr());
  osgWidget.SetColour(RavlImageN::RealRGBAValueC(1.0, 1.0, 1.0, 1.0));

  win.Add(vbox);

  Manager.Execute();

  win.Show();

  Manager.Wait();

  return 0;
}

RAVL_ENTRY_POINT(DoMain);
