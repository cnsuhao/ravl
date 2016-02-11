REQUIRES= CUDA

PACKAGE=Ravl/CUDA

HEADERS= Util.hh Array.hh ArrayOp.hh

SOURCES= Util.cc Array.cc ArrayOp.cu

PLIB= RavlCUDA

USESLIBS= CUDA RavlCore

MAINS= testRavlCUDA.cc

TESTEXES=
