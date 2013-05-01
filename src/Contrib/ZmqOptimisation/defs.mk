
DONOT_SUPPORT=VCPP

REQUIRES=ZeroMQ

PACKAGE= Ravl/Zmq

HEADERS= ZmqGeneticOptimiser.hh

SOURCES= ZmqGeneticOptimiser.cc

MUSTLINK=

PLIB= RavlZmqOptimisation

USESLIBS= RavlXMLFactory RavlZmq RavlService RavlRLog RavlGeneticOptimisation 

PROGLIBS= RavlIO RavlDPMT RavlOSIO RavlDPDisplay RavlExtImgIO RavlPatternRec

EXTERNALLIBS =

MAINS= exZmqGeneticOptimiser.cc doZmqOptimiserWorker.cc

AUXFILES= exZmqGeneticOptimiser.xml

AUXDIR=share/Ravl/Zmq

USESPKGCONFIG = 
