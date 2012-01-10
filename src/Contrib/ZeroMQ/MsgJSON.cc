
#include "Ravl/Zmq/MsgJSON.hh"
#include "Ravl/Zmq/MsgBuffer.hh"
#include "Ravl/DP/Converter.hh"

namespace RavlN { namespace ZmqN {

  //! Create a message buffer of a given size.
  MsgJSONC::MsgJSONC(size_t size)
    : SArray1dC<char>(MsgBufferC(size),size)
  {}

  SArray1dC<char> MsgJSON2SArray1d(const MsgJSONC &msg)
  { return msg; }

  DP_REGISTER_CONVERSION(MsgJSON2SArray1d,1.0);

  void LinkMsgJSON()
  {}
}}

