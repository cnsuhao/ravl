
PACKAGE=Ravl/CUDA

HEADERS= Util.hh Array.hh ArrayOp.hh

SOURCES= Util.cc Array.cc ArrayOp.cu

PLIB= RavlCUDA

USESLIBS= CUDA RavlRLog

USERNVCCFLAGS = -gencode arch=compute_10,code=sm_10 -gencode arch=compute_20,code=sm_20 

MAINS= testRavlCUDA.cc

TESTEXES=

EXTERNALLIBS=CUDABlas.def CUDARand.def 
