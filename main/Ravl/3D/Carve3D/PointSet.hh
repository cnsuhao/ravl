// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlCarve3D

#ifndef _POINTSETC_
#define _POINTSETC_

#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/RefCounter.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/3D/Vertex.hh"

namespace Ravl3DN
{
  using namespace RavlN;
  using namespace RavlImageN;

  class VoxelSetC;

  //!userlevel:Develop
  //:Array of 3D vertices and attributes

  class PointSetBodyC: public RCBodyVC
  {
  public:
    PointSetBodyC()
    {}
    //:Default constructor

    PointSetBodyC(UIntT numPoints,bool haveColour = false)
      : m_vertices(numPoints), m_haveColour(haveColour)
    { 
      if (m_haveColour) 
	m_colours = SArray1dC<ByteRGBValueC>(numPoints); 
    }
    //:Construct a point set with a number of points.

    PointSetBodyC(const SArray1dC<VertexC> &v, const SArray1dC<ByteRGBValueC> &col,bool haveColour = false)
      : m_vertices(v), m_colours(col), m_haveColour(haveColour)
    {}
    //:Construct a point set from an array of vertices and colours

    virtual ~PointSetBodyC()
    {}

  public:

    RCBodyVC& Copy() const;
    //:Make a copy of the set.
    
    SArray1dC<VertexC>& Vertices()
    { return m_vertices; }
    //:Access the array of all vertices in the point set.

    const SArray1dC<VertexC>& Vertices() const
    { return m_vertices; }
    //:Access the array of all vertices in the point set.
    
    SArray1dC<ByteRGBValueC>& Colours()
    { return m_colours; }
    //:Access the array of all colours in the point set.

    const SArray1dC<ByteRGBValueC>& Colours() const
    { return m_colours; }
    //:Access the array of all colours in the point set.

    bool HaveColour() const { return m_haveColour; }
    //:Do the points have colour attributes?

    Vector3dC Centroid() const;
    //:centroid of points

  protected:
    SArray1dC<VertexC> m_vertices;      // Array of point positions.
    SArray1dC<ByteRGBValueC> m_colours; // Array of colours.
    bool m_haveColour;                  // Indicates if points are coloured
  };

  //!userlevel:Normal
  //:Array of 3D vertices and attributes
  // Use Ravl-GUI-3D-DPointSetC to correctly render the point set

  class PointSetC: public RCHandleC<PointSetBodyC>
  {
  public:
    PointSetC()
    {}
    //:Default constructor

    PointSetC(UIntT numPoints,bool haveColour = false)
      : RCHandleC<PointSetBodyC>(*new PointSetBodyC(numPoints,haveColour))
    {}
    //:Construct a point set with a number of points.

  protected:
    PointSetC(PointSetBodyC &bod)
      : RCHandleC<PointSetBodyC>(bod)
    {}
    //:Body constructor.
    
    PointSetBodyC &Body()
    { return RCHandleC<PointSetBodyC>::Body(); }
    //:Access body.

    const PointSetBodyC &Body() const
    { return RCHandleC<PointSetBodyC>::Body(); }
    //:Access body.

  public:
    PointSetC Copy() const
    { return PointSetC(static_cast<PointSetBodyC &>(Body().Copy())); }
    //:Make a copy of the point set.

    bool HaveColour() const { return Body().HaveColour(); }
    //:Do the points have colour attributes?

    SArray1dC<VertexC>& Vertices()
    { return Body().Vertices(); }
    //:Access the array of all vertices in the point set.

    const SArray1dC<VertexC>& Vertices() const
    { return Body().Vertices(); }
    //:Access the array of all vertices in the point set.
    
    SArray1dC<ByteRGBValueC>& Colours()
    { return Body().Colours(); }
    //:Access the array of all colours in the point set.

    const SArray1dC<ByteRGBValueC>& Colours() const
    { return Body().Colours(); }
    //:Access the array of all colours in the point set.

    Vector3dC Centroid() const
      { return Body().Centroid(); }
    // Centroid of points
  };
  
}

#endif
