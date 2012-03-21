
namespace RavlN {
  extern void LinkDataServer();
  extern void LinkDataServerControlServer();

  void MustLinkRavlDataServer() {
    LinkDataServer();
    LinkDataServerControlServer();
  }

}
