PACKAGE = AVTools

HEADERS = MarkElem.hh

SOURCES = MarkElem.cc

MAINS= MarkSeq.cc 

USESLIBS=  RavlCore RavlDPMT RavlGUI RavlGUI2D RavlOS RavlThreads RavlVPlay

PROGLIBS=  RavlGUIUtil RavlIO RavlImage RavlImageProc RavlNet RavlOSIO \
RavlVideoIO RavlDPDisplay RavlImageIO \
RavlDV.opt RavlImgIOV4L.opt RavlExtImgIO.opt CSPDriver.opt RavlURLIO.opt \
RavlLibMPEG2.opt RavlImgIO1394dc.opt RavlDVDRead.opt RavlAVIFile.opt 


EHT= Markup.html
