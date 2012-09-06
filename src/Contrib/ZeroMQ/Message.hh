// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_ZMQN_MESSAGE_HEADER
#define RAVL_ZMQN_MESSAGE_HEADER
//! lib=RavlZmq

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

      //! Push a string.
      void Push(const RavlN::StringC &msg);

      //! Push a buffer onto the message stack
      void Push(const SArray1dC<char> &buff);

      //! Push contents of another message onto the end of this one.
      void Push(const MessageC &msg);

      //! Pop a buffer from the message stack
      void Pop(SArray1dC<char> &buff);

      //! Pop a message
      void Pop(std::string &str);

      //! Access next array to pop.
      SArray1dC<char> &Top()
      { return m_parts.back(); }

      //! Discard the top of the stack.
      void Pop()
      { m_parts.pop_back(); }

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
