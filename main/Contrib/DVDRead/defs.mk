REQUIRES = dvdread

PACKAGE = Ravl

HEADERS = DVDRead.hh DVDFormat.hh

SOURCES = DVDRead.cc DVDFormat.cc

PLIB = RavlDVDRead

EXAMPLES = testDVDRead.cc testDVDMPEG.cc

USESLIBS = RavlLibMPEG2 DVDRead RavlIO RavlCore

PROGLIBS = RavlDPDisplay RavlImage  

MUSTLINK = InitDVDFormat.cc

AUXFILES = DVDRead.def

AUXDIR = lib/RAVL/libdep
