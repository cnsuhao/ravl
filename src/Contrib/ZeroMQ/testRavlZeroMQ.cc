// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#include "Ravl/Zmq/Context.hh"
#include "Ravl/Zmq/Socket.hh"
#include "Ravl/Option.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/SysLog.hh"

using namespace RavlN::ZmqN;

int main(int nargs,char **argv) {
  RavlN::OptionC opts(nargs,argv);
  bool server = opts.Boolean("s",false,"Server");
  opts.Check();

  RavlN::SysLogOpen("testRavlZeroMQ");

  ContextC::RefT ctxt = new RavlN::ZmqN::ContextC(1);

  if(server) {
    SocketC::RefT skt = new SocketC(*ctxt,ZST_PUSH);
    skt->Bind("tcp://127.0.0.1:5551");
    for(int i = 0;i < 1000;i++) {
      RavlDebug("Sending Hello...");
      MessageC::RefT msg = new MessageC("Hello");
      skt->Send(*msg);
      RavlN::Sleep(0.2);
    }

  } else {
    SocketC::RefT skt = new SocketC(*ctxt,ZST_PULL);
    skt->Connect("tcp://127.0.0.1:5551");
//skt->Subscribe("");
    while(1) {
      MessageC::RefT msg;
      skt->Recieve(msg);
      std::string txt;
      msg->Pop(txt);
      RavlDebug("Got '%s' ",txt.data());
    }

  }


  return 0;
}
