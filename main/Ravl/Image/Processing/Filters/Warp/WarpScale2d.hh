// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_WARPSCALE2D_HEADER
#define RAVLIMAGE_WARPSCALE2D_HEADER 1
///////////////////////////////////////////////////////////////
//! docentry="Ravl.API.Images.Scaling and Warping"
//! lib=RavlImageProc
//! author="Charles Galambos"
//! rcsid="$Id$"
//! date="05/05/1999"
//! file="Ravl/Image/Processing/Filters/Warp/WarpScale2d.hh"

#include "Ravl/Image/Image.hh"
#include "Ravl/Image/BilinearInterpolation.hh"
#include "Ravl/Array2dIter.hh"
#include "Ravl/Vector2d.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/RealRange2d.hh"

#define CLEVER_BILINEAR
namespace RavlImageN {

  template <class InT, class OutT>
  bool WarpScaleBilinear(const ImageC<InT> &img,
                           const Vector2dC &scale, // Distance between samples in the input image.
                           ImageC<OutT> &result    // Output of scaling. The image must be of the appropriate size
                           )
  {
    //cout << "src frame:" << img.Frame() << std::endl;
    if(result.Frame().IsEmpty()) {
      const IndexRange2dC &imgFrame = img.Frame();
      IndexRange2dC rng;
      rng.TRow() = Ceil(imgFrame.TRow() / scale[0]);
      rng.LCol() = Ceil(imgFrame.LCol() / scale[1]);
#ifdef CLEVER_BILINEAR
      rng.BRow() = Floor((imgFrame.BRow() - 0) / scale[0]);
      rng.RCol() = Floor((imgFrame.RCol() - 0) / scale[1]);
#else
      rng.BRow() = Floor((imgFrame.BRow() - 1) / scale[0]);
      rng.RCol() = Floor((imgFrame.RCol() - 1) / scale[1]);
#endif
      result = ImageC<OutT>(rng);
    }
    //cout << "res frame:" << result.Frame() << std::endl;
    Point2dC origin(result.Frame().TRow() * scale[0], result.Frame().LCol() * scale[1]);
    //cout << "origin:" << origin << std::endl;

#if 0
    // Simple implementation.
    Point2dC rowStart = origin;
    for(Array2dIterC<OutT> it(result);it;) {
      Point2dC pnt = rowStart;
      do {
        BilinearInterpolation(img,pnt,*it);
        pnt[1] += scale[1];
      } while(it.Next()); // True while in same row.
      rowStart[0] += scale[0];
    }
#else
#ifdef CLEVER_BILINEAR
    Point2dC rowStart = origin;
    for(Array2dIterC<OutT> it(result);it;) {
      Point2dC pnt = rowStart;

      IntT fx = Floor(pnt[0]); // Row
      RealT u = pnt[0] - fx;
      if(u < 1e-5) {
        do {
          IntT fy = Floor(pnt[1]); // Col
          RealT t = pnt[1] - fy;
          if(t < 1e-5) {
            const InT* pixel1 = &(img)[fx][fy];
            *it = OutT(pixel1[0]);
            pnt[1] += scale[1];
          } else {
            RealT onemt = (1.0-t);

            //printf("x:%g  y:%g  fx:%i  fy:%i\n", pnt[0], pnt[1], fx, fy);
            const InT* pixel1 = &(img)[fx][fy];
            *it = OutT((pixel1[0] * onemt) +
                        (pixel1[1] * t));
            pnt[1] += scale[1];
          }
        } while(it.Next()); // True while in same row.
      } else {
        RealT onemu = (1.0-u);
        do {
          IntT fy = Floor(pnt[1]); // Col
          RealT t = pnt[1] - fy;
          if(t < 1e-5) {
            const InT* pixel1 = &(img)[fx][fy];
            const InT* pixel2 = &(img)[fx+1][fy];
            *it = OutT((pixel1[0] * onemu) +
                        (pixel2[0] * u));
            pnt[1] += scale[1];
          } else {
            RealT onemt = (1.0-t);

            //printf("x:%g  y:%g  fx:%i  fy:%i\n", pnt[0], pnt[1], fx, fy);
            const InT* pixel1 = &(img)[fx][fy];
            const InT* pixel2 = &(img)[fx+1][fy];
            *it = OutT((pixel1[0] * (onemt*onemu)) +
                        (pixel1[1] * (t*onemu)) +
                        (pixel2[0] * (onemt*u)) +
                        (pixel2[1] * (t*u)));
            pnt[1] += scale[1];
          }
        } while(it.Next()); // True while in same row.
      }

      rowStart[0] += scale[0];
    }
#else
    Point2dC rowStart = origin;
    for(Array2dIterC<OutT> it(result);it;) {
      Point2dC pnt = rowStart;

      IntT fx = Floor(pnt[0]); // Row
      RealT u = pnt[0] - fx;
      RealT onemu = (1.0-u);
      do {
        IntT fy = Floor(pnt[1]); // Col
        RealT t = pnt[1] - fy;
        RealT onemt = (1.0-t);

        //printf("x:%g  y:%g  fx:%i  fy:%i\n", pnt[0], pnt[1], fx, fy);
        const InT* pixel1 = &(img)[fx][fy];
        const InT* pixel2 = &(img)[fx+1][fy];
        *it = OutT((pixel1[0] * (onemt*onemu)) +
                    (pixel1[1] * (t*onemu)) +
                    (pixel2[0] * (onemt*u)) +
                    (pixel2[1] * (t*u)));
        pnt[1] += scale[1];
      } while(it.Next()); // True while in same row.

      rowStart[0] += scale[0];
    }
#endif
#endif
    return true;
  }
  //: Scale an image
}

#endif
