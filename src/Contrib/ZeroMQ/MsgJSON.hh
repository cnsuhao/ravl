#ifndef RAVL_ZMQ_MSGJSON_HH_
#define RAVL_ZMQ_MSGJSON_HH_

#include "Ravl/SArray1d.hh"

namespace RavlN { namespace ZmqN {

  //! Message coded in JSON

  class MsgJSONC
    : public SArray1dC<char>
  {
  public:
    //! Create an empty buffer.
    MsgJSONC()
    {}

    //! Create a message buffer of a given size.
    MsgJSONC(size_t size);

    //! Constructor
    MsgJSONC(const SArray1dC<char> &array)
     : SArray1dC<char>(array)
    {}

  };

}}

#endif /* MSGJSON_HH_ */
