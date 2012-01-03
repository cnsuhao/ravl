#ifndef RAVL_ZMQN_MESSAGE_HEADER
#define RAVL_ZMQN_MESSAGE_HEADER

#include "Ravl/Zmq/Context.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/String.hh"
#include <vector>
#include <string>

namespace RavlN {
  namespace ZmqN {

    //! Multi-part message.

    class MessageC
     : public RCBodyC
    {
    public:
      //! Construct an empty message.
      MessageC();

      //! Construct a message from some data
      MessageC(const SArray1dC<char> &data);

      //! Construct a message from a string
      MessageC(const std::string &data);

      //! Construct a message from a string
      MessageC(const char *data);

      //! Destructor.
      ~MessageC();

      //! Push a string.
      void Push(const char *msg);

      //! Push a string.
      void Push(const std::string &msg);

      //! Push a buffer onto the message stack
      void Push(const SArray1dC<char> &buff);

      //! Push contents of another message onto the end of this one.
      void Push(const MessageC &msg);

      //! Pop a buffer from the message stack
      void Pop(SArray1dC<char> &buff);

      //! Pop a message
      void Pop(std::string &str);


      //! Access parts
      std::vector<SArray1dC<char> > &Parts()
      { return m_parts; }

      //! Access parts
      const std::vector<SArray1dC<char> > &Parts() const
      { return m_parts; }

      //! Handle to message.
      typedef SmartPtrC<MessageC> RefT;
    protected:
      std::vector<SArray1dC<char> > m_parts;
    };
  }
}


#endif
