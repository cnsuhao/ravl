
REQUIRES=iksemel

PACKAGE=Ravl/XMPP

HEADERS=IksemelConnection.hh

SOURCES=IksemelConnection.cc

PLIB=RavlXMPPIksemel

MAINS= testIksemel.cc

USESLIBS=RavlOS iksemel RavlXMPP

EXTERNALLIBS=iksemel.def

CCPPFLAGS += -DRLOG_COMPONENT=Ravl
