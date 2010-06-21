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
#include "Ravl/OpenSceneGraph/TexTriMesh.hh"
#include "Ravl/OpenSceneGraph/HUD.hh"
#include "Ravl/3D/TexTriMesh.hh"
#include "Ravl/Vector3d.hh"
#include "Ravl/3D/Vertex.hh"
#include "Ravl/3D/Tri.hh"

using namespace RavlN;
using namespace RavlGUIN;
using namespace RavlOSGN;

ImageByteRGBAC::RefT CreateImage()
{
  ImageByteRGBAC::RefT imageRef = new ImageByteRGBAC(RealRange2dC(-1, 1));
  ImageC<ByteRGBAValueC> image;
  if (Load(PROJECT_OUT "/share/RAVL/pixmaps/monkey.ppm", image, "", true))
    imageRef->SetImage(image);

  return imageRef;
}

TexTriMeshC::RefT CreateTexTriMesh()
{
  TexTriMeshC::RefT texTriMeshRef = new TexTriMeshC;

  SArray1dC<Ravl3DN::VertexC> vertexArray(4);
  vertexArray[0] = Ravl3DN::VertexC(Point3dC(0, 0, 0));
  vertexArray[1] = Ravl3DN::VertexC(Point3dC(1, 0, 0));
  vertexArray[2] = Ravl3DN::VertexC(Point3dC(1, -1, 0));
  vertexArray[3] = Ravl3DN::VertexC(Point3dC(0, -1, 0));

  TFVectorC<ByteT, 3> colourWhite(3);
  colourWhite[0] = 255;
  colourWhite[1] = 255;
  colourWhite[2] = 255;

  SArray1dC<Ravl3DN::TriC> triArray(2);

  triArray[0] = Ravl3DN::TriC(vertexArray[0],
                              vertexArray[1],
                              vertexArray[2],
                              Point2dC(0, 0),
                              Point2dC(1, 0),
                              Point2dC(1, 1),
                              0);
  triArray[0].SetFaceNormal(Vector3dC(0, 0, 1));
  triArray[0].SetColour(colourWhite);

  triArray[1] = Ravl3DN::TriC(vertexArray[0],
                              vertexArray[2],
                              vertexArray[3],
                              Point2dC(0, 0),
                              Point2dC(1, 1),
                              Point2dC(0, 1),
                              0);
  triArray[1].SetFaceNormal(Vector3dC(0, 0, 1));
  triArray[1].SetColour(colourWhite);

  SArray1dC<ImageC<ByteRGBValueC> > imageArray(1);
  if (!Load(PROJECT_OUT "/share/RAVL/pixmaps/monkey.ppm", imageArray[0], "", true))
    return TexTriMeshC::RefT();

  SArray1dC<StringC> filenameArray(1);
  filenameArray[0] = PROJECT_OUT "/share/RAVL/pixmaps/monkey.ppm";
  
  Ravl3DN::TexTriMeshC texTriMesh(vertexArray, triArray, imageArray, filenameArray);

  texTriMeshRef->SetMesh(texTriMesh);

  return texTriMeshRef;
}

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
  OpenSceneGraphWidgetC osgWidget(400, 400);
  Connect(osgWidget.Signal("button_press_event"), &pressFunc);
  Connect(osgWidget.Signal("button_release_event"), &releaseFunc);

  RavlGUIN::LBoxC vbox = VBox(osgWidget);

  // Root object
  GroupC::RefT groupRef = new GroupC;

  // Add a sphere
  TransformPositionAttitudeC::RefT transformSphereRef = new TransformPositionAttitudeC;
  groupRef->AddChild(transformSphereRef.BodyPtr());

  SphereC::RefT sphereRef = new SphereC;
  sphereRef->SetColour(RavlImageN::RealRGBAValueC(0.5, 0.0, 0.0, 1.0));

  GeodeC::RefT geodeRef = new GeodeC;
  geodeRef->AddDrawable(sphereRef.BodyPtr());

  transformSphereRef->AddChild(geodeRef.BodyPtr());

  // Add an image
  TransformPositionAttitudeC::RefT transformImageRef = new TransformPositionAttitudeC;
  transformImageRef->SetPosition(Vector3dC(1, 0, 0));
  transformImageRef->SetAttitude(Quatern3dC(Vector3dC(1, 0, 0), RavlConstN::pi / 2.0));
  groupRef->AddChild(transformImageRef.BodyPtr());

  ImageByteRGBAC::RefT imageRef = CreateImage();
  transformImageRef->AddChild(imageRef.BodyPtr());

  vbox.Add(Button("Change Image", &CBSetImage, imageRef));

  // Add some text
  TransformPositionAttitudeC::RefT transformTextRef = new TransformPositionAttitudeC;
  transformTextRef->SetPosition(Vector3dC(1.5, 0, 0));
  groupRef->AddChild(transformTextRef.BodyPtr());

  TextC::RefT textRef = new TextC("Some Text");
  textRef->SetSize(0.5);
  textRef->SetAlignment(TextAlignCentreBaseLine);
  textRef->SetAxisAlignment(TextAxisScreen);
  textRef->SetColour(RavlImageN::RealRGBAValueC(0.5, 0.5, 0.5, 1.0));

  GeodeC::RefT textGeodeRef = new GeodeC;
//  textGeodeRef->BringToFront();
  textGeodeRef->AddDrawable(textRef.BodyPtr());

  transformTextRef->AddChild(textGeodeRef.BodyPtr());

  // Add a TriMesh
  TransformPositionAttitudeC::RefT transformTexTriMeshRef = new TransformPositionAttitudeC;
  transformTexTriMeshRef->SetPosition(Vector3dC(-2, 0, 0));
  transformTexTriMeshRef->SetAttitude(Quatern3dC(Vector3dC(1, 0, 0), RavlConstN::pi / 2.0));
  groupRef->AddChild(transformTexTriMeshRef.BodyPtr());

  TexTriMeshC::RefT texTriMeshRef = CreateTexTriMesh();
  transformTexTriMeshRef->AddChild(texTriMeshRef.BodyPtr());
  
  HUDC::RefT hudRef = new HUDC(RealRange2dC(0, 768, 0, 1024));
  groupRef->AddChild(hudRef.BodyPtr());

  TextC::RefT textHUDRef = new TextC("HUUUD");
//  textHUDRef->SetFont("/usr/share/fonts/truetype/DejaVuSans.ttf");
  textHUDRef->SetSize(100);
  textHUDRef->SetPosition(Vector3dC(100, 100, -1));
  textHUDRef->SetAlignment(TextAlignCentreBaseLine);
  textHUDRef->SetAxisAlignment(TextAxisScreen);
  textHUDRef->SetColour(RavlImageN::RealRGBAValueC(0.0, 0.0, 0.0, 1.0));

  GeodeC::RefT textHUDGeodeRef = new GeodeC;
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
