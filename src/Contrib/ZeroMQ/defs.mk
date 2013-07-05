# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2011, Charles Galambos
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! author=Charles Galambos
#! docentry=Ravl.API.ZeroMQ

DONOT_SUPPORT=VCPP

REQUIRES=ZeroMQ

PACKAGE= Ravl/Zmq

HEADERS= Context.hh Message.hh Socket.hh MsgBuffer.hh SocketDispatcher.hh Reactor.hh \
 SocketDispatchTrigger.hh MsgJSON.hh MsgSmartPtr.hh

SOURCES= ZmqContext.cc ZmqMessage.cc ZmqSocket.cc ZmqMsgBuffer.cc ZmqSocketDispatcher.cc ZmqReactor.cc \
 ZmqSocketDispatchTrigger.cc ZmqMsgJSON.cc ZmqMsgSmartPtr.cc

MUSTLINK=LinkRavlZeroMQ.cc

PLIB= RavlZmq

USESLIBS=RavlCore RavlXMLFactory Zmq RavlService

EXTERNALLIBS = Zmq.def

MAINS= testRavlZeroMQ.cc testRavlZeroMQXML.cc

