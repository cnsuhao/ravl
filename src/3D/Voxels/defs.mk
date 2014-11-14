# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006-14, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
DESCRIPTION = Voxel processing library

PACKAGE = Ravl/Voxels

PLIB = RavlVoxels

SUMMARY_LIB=Ravl

HEADERS = \
 Ray3d.hh \
 VoxelGrid.hh \
 VoxeliseMesh.hh \
 VoxelOctree.hh \
 VoxelSet.hh \
 VoxelSetIter.hh \
 VoxelRayIter.hh \
 VoxelSkeleton.hh \
 VoxelSurface.hh

SOURCES = \
 VoxeliseMesh.cc \
 VoxelRayIter.cc \
 VoxelSkeleton.cc \
 VoxelSurface.cc 


USESLIBS = Ravl3D

