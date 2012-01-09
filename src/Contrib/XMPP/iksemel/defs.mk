
REQUIRES=iksemel RLog

PACKAGE=Ravl/XMPP

HEADERS=IksemelConnection.hh

SOURCES=IksemelConnection.cc

PLIB=RavlXMPPIksemel

MAINS= testIksemel.cc

USESLIBS=RavlOS iksemel RavlXMPP

PROGLIBS=RavlIO RavlXMLFactory

MUSTLINK=LinkRavlXMPPIksemel.cc

EXTERNALLIBS=iksemel.def

CCPPFLAGS += -DRLOG_COMPONENT=Ravl
