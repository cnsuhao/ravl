SOURCES=Dummy.cc

PLIB=RavlLibManager

LIBS=$(ALL_EXTERNAL_LIBS)

PLIBDEPENDS=$(patsubst lib%,%,$(SINGLE_RECIPE))

SINGLESO=libRavl
