REQUIRES = libmpeg2

PACKAGE = Ravl

HEADERS = DVDRead.hh

SOURCES = DVDRead.cc

PLIB = RavlDVDRead

USESLIBS = DVDRead RavlDPDisplay RavlImage RavlLibMPEG2 RavlIO RavlCore

EXAMPLES = testDVDRead.cc testDVDMPEG.cc

AUXFILES = DVDRead.def
