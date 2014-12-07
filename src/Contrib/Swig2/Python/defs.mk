# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

REQUIRES=Swig SwigPython NumPy

PACKAGE=Ravl/Swig

HEADERS=

SOURCES=Ravl.i

PLIB=RavlPythonSwig2

SINGLESO=_RavlSwigPython2

USESLIBS=RavlDPDisplay.opt RavlGUI2D.opt RavlGUI.opt RavlPatternRecIO RavlSVM \
RavlGnuPlot RavlPatternRec RavlDataSet fann MathIO RavlMath ccmath \
RavlExtImageIO RavlImageIO RavlImage RavlNumPy NumPy Python RavlNet \
RavlService RavlXMLFactory RavlThreads RavlOSIO RavlOS RavlIO RavlCore

PROGLIBS=RavlMath

SWIGOPTS=-python

AUXDIR=/share/Python

AUXFILES=Ravl.py

SCRIPTS= test.py testVectorMatrix.py testPatternRec.py testSVM.py \
 testReduce.py testFeatureSelect.py testNumPy.py
