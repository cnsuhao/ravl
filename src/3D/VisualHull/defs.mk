# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
DESCRIPTION = Visual hull library

PACKAGE = Ravl/VisualHull

PLIB = RavlVisualHull

HEADERS = \
 MovingLeastSquares.hh \
 VisualHull.hh \
 VoxelCarve.hh \
 VoxelClean.hh \
 VoxelColourCarve.hh \
 VoxelDistanceTransform.hh

# PhantomVolume.hh 

SOURCES = \
 MovingLeastSquares.cc \
 VisualHull.cc \
 VoxelCarve.cc \
 VoxelClean.cc \
 VoxelColourCarve.cc \
 VoxelDistanceTransform.cc

# PhantomVolume.cc 


USESLIBS = RavlVoxels
