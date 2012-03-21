#include "WhiteLineDetector.hh"
#include "Ravl/IO.hh"
#include "Ravl/LinePP2d.hh"
#include "Ravl/Image/DrawLine.hh"
#include "Ravl/Image/DrawCircle.hh"
#include "Ravl/Image/Font.hh"
#include "Ravl/Image/ImageConv.hh"

using namespace RavlN;
using namespace RavlImageN;
using namespace RavlConstN;


WhiteLineDetectorBodyC::WhiteLineDetectorBodyC(bool Verbose)
  : darkred(128,   0,   0),
    red    (255,   0,   0),
    yellow (255, 255,   0),
    brown  (128, 128,   0),
    verbose(Verbose),
    gFilter(3),
    edgeDet(true, 0.3, 1.0),
    pphtProc(Point2dC(0.01, 1.0),0.9999999,90,-1,false,1.4,true),
    maxAngle(1), maxSep(6), overlap(0.5), minLen(50)
{
  pphtProc.SetCorridor(4);
  pphtProc.SetMaxGap(25);
  pphtProc.SetMinLineLength(minLen);
  pphtProc.SetMagSort(false);
  pphtProc.SetDoSHT(false);
}

SArray1dC<LinePP2dC> WhiteLineDetectorBodyC::Apply(const ImageC<RealT> &img) {
  ImageC<ByteRGBValueC> canvas = RealRGBImageCT2ByteRGBImageCT(RealImageCT2RealRGBImageCT(img));
  ImageC<RealT> fImg = gFilter.Apply(img);
  if (verbose)  RavlN::Save("@X:filter", fImg);

  // Find straight edges
  DListC<EdgelC> inList = edgeDet.LApply(fImg);
  PCPixMappingC<Curve2dLineSegmentC> ht = pphtProc.Apply(inList);
  DListC<LinePP2dC> lines;
  for (DLIterC<PCPixMapC<Curve2dLineSegmentC> >  it(ht);it;it++) {
    LinePP2dC line(it->Curve().StartPnt(),it->Curve().EndPnt());
    if (line.Length() > minLen) {
        lines.InsLast(line);
        DrawLine(canvas, yellow, line);
        DrawCircle(canvas, red, line.P1(), 1);
    }
    else DrawLine(canvas, brown, line);
  }

  // Find suitable-looking pairs of edges & save their mean as ridge
  DListC<LinePP2dC> ridges;
  for (DLIterC<LinePP2dC> l(lines); l; l++) {
    DLIterC<LinePP2dC> m(l);
    m++;
    for (; m; m++) {
      RealT signedGap = l->SignedDistance(m->MidPoint());  // ridges, not troughs
      if (// parallel & opposite dirns & overlapping
          (l->UnitNormal().Dot(m->UnitNormal()) < -cos(RavlConstN::pi/180.0*maxAngle)) &&
          // "clockwise pair" && close by
          (signedGap > -maxSep) && (signedGap < 0.0) &&
          // overlapping one way or another
          (((l->ParClosest(m->P1()) > overlap) && (l->ParClosest(m->P2()) < 1.0-overlap))) &&
          (((m->ParClosest(l->P1()) > overlap) && (m->ParClosest(l->P2()) < 1.0-overlap)))
          ) {
        // create ridge as "mean" line from pair
        LinePP2dC ridge(Point2dC((m->P1()+l->P2())/2.0),Point2dC((l->P1()+m->P2())/2.0));
        // orientate it consistently
        if (ridge[0].Sum() > ridge[1].Sum())  ridge = ridge.Swapped();
        ridges.InsLast(ridge); // add to list
        DrawLine(canvas, darkred, ridge);
        //DrawText(DefaultFont(), red, ridge.MidPoint(), StringC(l->UnitNormal().Dot(m->UnitNormal())*180.0/RavlConstN::pi), canvas);
        // remove pair of edge lines from further consideration
        m.Del();
        break;
      }
    }
  }

  // Every valid court candidate must intersect as T or L with at least one other candidate
  // This means, for each line A find another line B that:
  //   - intersects with A at "right angles", and
  //   - intersects with A "within" B, and
  //   - one of A's end points is "near" B or they intersect "within" A
  for (DLIterC<LinePP2dC> r(ridges); r; r++) {
    bool intersectFound(false);
    for (DLIterC<LinePP2dC> s(ridges); s; s++) {
      RealT p1 = s->ParIntersection(*r); // point on s where r intersects with s
      RealT p2 = r->ParIntersection(*s); // point on r where r intersects with s
      if ((Abs(r->UnitNormal().Dot(s->UnitNormal())) < cos(pi/6.0)) &&
          ((p1 > -0.02) && (p1 < 1.02)) &&
          ((p2 > -0.02) && (p2 < 1.02))
          ) {
        intersectFound = true;
        //cout<<"Line "<<i++<<"\nr "<<*r<<"\ns "<<*s<<"\n"<<p1<<" "<<p2<<endl;
        break;
      }
    }
    if (!intersectFound) r.Del();
  }
  for (DLIterC<LinePP2dC> r(ridges); r; r++)  DrawLine(canvas, red, *r);
  if (verbose)  RavlN::Save("@X:lines", canvas);

  SArray1dC<LinePP2dC> ridgeArray(ridges.Size());
  for (SArray1dIterC<LinePP2dC>i(ridgeArray); i; ++i) 
    *i = ridges.PopFirst();

  return ridgeArray;
}
