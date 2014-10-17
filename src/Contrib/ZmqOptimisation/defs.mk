
DONOT_SUPPORT=VCPP

REQUIRES=ZeroMQ RLog

PACKAGE= Ravl/Zmq

HEADERS= ZmqGeneticOptimiser.hh

SOURCES= ZmqGeneticOptimiser.cc

MUSTLINK=

PLIB= RavlZmqOptimisation

USESLIBS= RavlXMLFactory RavlZmq RavlService RavlRLog RLog RavlGeneticOptimisation 
# including Ravl/RLog.hh in the source causes a dependency on both RavlRLog
# and RLog itself.

PROGLIBS= RavlIO RavlDPMT RavlOSIO RavlDPDisplay.opt RavlExtImgIO RavlPatternRec

EXTERNALLIBS =

MAINS= exZmqGeneticOptimiser.cc doZmqOptimiserWorker.cc

AUXFILES= exZmqGeneticOptimiser.xml

AUXDIR=share/Ravl/Zmq

USESPKGCONFIG = 
