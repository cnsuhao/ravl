# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2002, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"
#! file="Ravl/3D/Mesh/defs.mk"

PACKAGE=Ravl/3D

HEADERS= Vertex.hh Tri.hh TriMesh.hh TexTriMesh.hh\
 HEMeshVertex.hh HEMeshEdge.hh HEMeshFace.hh HEMesh.hh \
 HEMeshVertexIter.hh  HEMeshFaceIter.hh VertexColourByteRGB.hh \
 MeshShapes.hh BuildTexTriMesh.hh SurfacePoint3dArray.hh


SOURCES= Vertex.cc Tri.cc TriMesh.cc TriMeshBinIO.cc \
 TexTriMesh.cc TexTriMeshBinIO.cc \
 HEMeshVertex.cc HEMeshEdge.cc HEMeshFace.cc HEMesh.cc \
 TriMesh2HEMesh.cc HEMeshFaceIter.cc VertexColourByteRGB.cc \
 MeshShapes.cc BuildTexTriMesh.cc SurfacePoint3dArray.cc

TESTEXES=testHEMesh.cc testTriMesh.cc

EXAMPLES = exBuildMesh.cc

PLIB=Ravl3D

SUMMARY_LIB=Ravl

USESLIBS=RavlMath RavlImage

PROGLIBS=RavlDPDisplay.opt RavlDPDisplay3d.opt

EHT=Ravl.API.3D.Mesh.html

