
#include "Ravl/Option.hh"
#include "Ravl/XMPP/IksemelConnection.hh"
#include "Ravl/RLog.hh"
#include "Ravl/OS/Date.hh"

using RavlN::StringC;

int main(int nargs,char **argv)
{
  RavlN::OptionC opts(nargs,argv);
  StringC id = opts.String("id","test@jabber.org","Connection id");
  opts.Check();
  RavlN::RLogInit(true);
  using RavlN::XMPPN::IksemelConnectionC;
  
  IksemelConnectionC::RefT con = new IksemelConnectionC("jabber.org","omnitest","wateshouse","default");

  con->Open("omnitest@jabber.org/cento","wateshouse");

  while(!con->IsReady())
    RavlN::Sleep(1);
  
  con->SendText("craftit@jabber.org","Hello!");
  while(1)
    RavlN::Sleep(1);

  return 0;
}


