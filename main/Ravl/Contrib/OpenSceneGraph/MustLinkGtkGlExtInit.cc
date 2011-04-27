
namespace RavlOSGN
{

  extern bool RegisterGtkGlExtInit();
  extern bool LinkOpenSceneGraphWidget();
  extern void LinkImageByteRGBA();
  extern void LinkImageByteRGB();
  extern void LinkSphere();
  extern void LinkGroup();
  extern void LinkText();
  extern void LinkTransform();
  extern void LinkHUD();
  extern void LinkBox();
  extern void LinkModelFile();
  extern void LinkLayout();
  extern void LinkLayoutStack();
  extern void LinkLayoutGrid();
  
  void LinkGtkGlExtInit()
  {
    RegisterGtkGlExtInit();
    LinkOpenSceneGraphWidget();
    
    LinkImageByteRGBA();
    LinkImageByteRGB();
    LinkSphere();
    LinkGroup();
    LinkText();
    LinkTransform();
    LinkHUD();
    LinkBox();
    LinkModelFile();
    LinkLayout();
    LinkLayoutStack();
    LinkLayoutGrid();
  }
}
