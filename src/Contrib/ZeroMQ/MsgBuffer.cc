
#include "Ravl/Zmq/MsgBuffer.hh"

namespace RavlN {
  namespace ZmqN {


    MsgBufferBodyC::MsgBufferBodyC(zmq_msg_t &msg)
     : RavlN::BufferBodyC<char>()
    {
      zmq_msg_init(&m_msg);
      zmq_msg_move(&m_msg,&msg);
      Build();
    }

    //! Create a buffer of a given size
    MsgBufferBodyC::MsgBufferBodyC(size_t size)
    {
      if(size > 0) {
        zmq_msg_init_size (&m_msg,size);
        Build();
      } else {
        zmq_msg_init(&m_msg);
        this->sz = 0;
      }
    }


    MsgBufferBodyC::MsgBufferBodyC()
    {
      zmq_msg_init(&m_msg);
      this->sz = 0;
    }

    MsgBufferBodyC::~MsgBufferBodyC() {
      zmq_msg_close(&m_msg);
    }

    void MsgBufferBodyC::Build()
    {
      this->buff = reinterpret_cast<char *>(zmq_msg_data (&m_msg));
      this->sz = zmq_msg_size(&m_msg);
    }

    static void FreeArrayFunc(void *data,void *hint) {
      SArray1dC<char> *arr = reinterpret_cast<SArray1dC<char> *>(hint);
      delete arr;
    }


    void ArrayToMessage(zmq_msg_t &msg,const SArray1dC<char> &arr) {
      // Is it a message already ?
      MsgBufferC buf(arr.Buffer());
      if(buf.IsValid()) {
        zmq_msg_init(&msg);
        zmq_msg_copy(&msg,buf.Msg());
        return ;
      }

      // Link msg to array.
      SArray1dC<char> *hint = new SArray1dC<char>(arr);
      size_t elemSize = hint->Size().V();
      zmq_msg_init_data (&msg, hint->ReferenceElm(),elemSize, &FreeArrayFunc, (void *) hint);
    }


  }
}
