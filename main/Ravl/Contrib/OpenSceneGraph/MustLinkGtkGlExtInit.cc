
namespace RavlOSGN
{

  extern bool RegisterGtkGlExtInit();
  extern bool LinkOpenSceneGraphWidget();

  void LinkGtkGlExtInit()
  {
    RegisterGtkGlExtInit();
    LinkOpenSceneGraphWidget();
  }
}
