PACKAGE = Ravl/Image

REQUIRES = OpenCV

MAINS =  exOpenCV.cc

HEADERS = OpenCVConvert.hh

SOURCES = OpenCVConvert.cc

PLIB = RavlOpenCV

EXTERNALLIBS = OpenCV.def

USESLIBS = RavlImage OpenCV 

PROGLIBS = RavlImageIO RavlDPDisplay 

TESTEXES = testOpenCV.cc

EXAMPLES =  exOpenCV.cc
