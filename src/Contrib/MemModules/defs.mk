PACKAGE = Ravl/Threads

HEADERS = MemModule.hh MemModules.hh MemItem.hh \
          MemUnit.hh MemIter.hh
SOURCES = MemModule.cc MemModules.cc MemItem.cc \
          MemUnit.cc MemIter.cc 

PLIB = RavlMemModules
 
USESLIBS=  RavlCore RavlImage RavlOS RavlThreads RavlImageIO

EHT = ThreadedMemory.html Modules_Header.html Wrapper.html ModuleSet.html MainProg.html Internals.html Example.html

AUXDIR = share/doc/RAVL/Images
AUXFILES = ModuleGraph.pdf Memory.pdf


TESTEXES = testMemModules.cc
SCRIPTS = MemoryMove

PROGLIBS = RavlExtImgIO

EXAMPLES = testMemModules.cc
