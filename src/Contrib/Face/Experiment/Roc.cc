// This file is part of OmniSoft, Pattern recognition software 
// Copyright (C) 2003, Omniperception Ltd.
// file-header-ends-here
////////////////////////////////////////////////////////////////////////////
//! author="Kieron J Messer"
//! lib=OmniFaceXml
//! date="11/10/101"
//! file="OmniSoft/Biometrics/TestHarness/Roc.cc"

#include "Ravl/Face/Roc.hh"
#include "Ravl/CDLIter.hh"
#include "Ravl/Assert.hh"
#include "Ravl/DP/FileFormatStream.hh"
#include "Ravl/DP/FileFormatBinStream.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/SArray1dIter.hh"
#include "Ravl/MeanVariance.hh"
#include "Ravl/OS/Filename.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {
  namespace FaceN {

    using namespace RavlGUIN;

    bool compare_claim_ascend(const ClaimT & el1, const ClaimT & el2)
    {
      return (el1.Data2() < el2.Data2());
    }

    bool compare_claim_descend(const ClaimT & el1, const ClaimT & el2)
    {
      return (el1.Data2() > el2.Data2());
    }

    RocBodyC::RocBodyC(bool highM) :
        highMatches(highM)
    {
      impostors = 0;
      clients = 0;
    }

// RocBodyC ///////////////////////////////////////////////////
//: Constructor
    RocBodyC::RocBodyC(const DListC<RankT> & roc, UIntT nclients, UIntT nimpostors, bool highM)
    {
      highMatches = highM;
      clients = nclients;
      impostors = nimpostors;
      for (ConstDLIterC<RankT> it(roc); it; it++) {
        ClaimT claim;
        if (it.Data().Data1() == "client")
          claim = ClaimT(true, it.Data().Data2());
        else
          claim = ClaimT(false, it.Data().Data2());
        claims.InsLast(claim);
      }
      //: Sort scores according to metric type
      Sort();
    }

    RocBodyC::RocBodyC(const ScoreSessionC & session, const StringC & type, bool highM) :
        metricType(type), highMatches(highM)
    {
      impostors = 0;
      clients = 0;
      //: Insert all relevant scores into the list
      for (ConstDLIterC<ScoreC> it(session); it; it++) {
        RealT meas = it.Data().Measure(type);
        ClaimT claim;
        if (it.Data().ClaimedId() == it.Data().ActualId()) {
          claim = ClaimT(true, meas); // true claim
          clients++;
        } else {
          claim = ClaimT(false, meas); // impostor attack
          impostors++;
        }
        claims.InsLast(claim);
      }
      Sort();
    }

    RocBodyC::RocBodyC(RealT max, RealT min, bool highM) :
        metricType("expert_0"), highMatches(highM)
    {
      impostors = 0;
      clients = 0;
      RealT t = (max - min) / 2.0;
      RealT step = (max - min) / 100.0;
      for (RealT v = min; v <= max; v += step) {
        ClaimT claim;
        if (v < t) {
          claim = ClaimT(false, v);
          impostors++;
        } else {
          claim = ClaimT(true, v);
          clients++;
        }
        claims.InsLast(claim);
      }

      //: Sort scores according to metric type
      Sort();
    }

//: Stream constructor
// This creates a new instance of the class from an input stream.

    RocBodyC::RocBodyC(istream &in)
    {
      in >> claims;
      in >> metricType;
      in >> highMatches;
      in >> clients;
      in >> impostors;
    }

    RocBodyC::RocBodyC(BinIStreamC &in)
    {
      in >> claims;
      in >> metricType;
      in >> highMatches;
      in >> clients;
      in >> impostors;
    }

//: Save to ostream.
    
    bool RocBodyC::Save(ostream &out) const
    {
      out << (*this);
      return true;
    }

    bool RocBodyC::Save(BinOStreamC &out) const
    {
      out << (*this);
      return true;
    }
    
    void RocBodyC::Sort()
    {
      if (highMatches) {
        claims.MergeSort(compare_claim_descend);
      } else
        claims.MergeSort(compare_claim_ascend);
      sorted = true;
    }
    
    void RocBodyC::Insert(const RealT & score, bool client)
    {
      sorted = false;
      ClaimT claim;
      if (client) {
        claim = ClaimT(true, score); // true claim
        clients++;
      } else {
        claim = ClaimT(false, score); // impostor attack
        impostors++;
      }
      claims.InsLast(claim);
    }
    
    RealT RocBodyC::EqualErrorRate() const
    {

      //: Check it is sorted before use
      if (!sorted)
        return -1.0;

      if (!IsValid())
        return -1.0;

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;
      RealT minSoFar = RavlConstN::maxReal;
      RealT eer = 0.0;
      RealT thresh = 0.0;
      RealT prevScore = 0.0;
      bool first = true;

      for (ConstDLIterC<ClaimT> it(claims); it; it++) {

        RealT currentScore = it.Data().Data2();

        // If first claim then we set the threshold to this
        if (first) {
          first = false;
          prevScore = currentScore;
          thresh = currentScore;
        }

        if (it.Data().Data1()) {
          clientssofar += 1.0;
        }
        else {
          impostorssofar += 1.0;
        }

        // Work out errors at this point on the ROC
        RealT clientsrejected = clients - clientssofar;
        RealT impostorsaccepted = impostorssofar;
        RealT fa = 0.0;
        RealT fr = 0.0;
        if (impostors > 0) {
          fa = impostorsaccepted / impostors;
        }
        if (clients > 0) {
          fr = clientsrejected / clients;
        }

        // Compute the difference between fr and fa
        // When this is a minimum then we can say that
        // the error rates are equal(ish).
        //RavlInfo("FA %0.4f FR %0.4f Score %0.4f", fa, fr, currentScore);
        RealT dif = Abs(fr - fa);

        // If we have a new minimum we update the threshold
        // where this occurs.
        if (dif < minSoFar) {

          // we will only update things if the score has changed.
          // Some classifiers do no output smooth ranges!
          if (currentScore == prevScore) {
            continue;
          }

          minSoFar = dif;
          eer = (fr + fa) / 2.0;
          thresh = (prevScore + currentScore) / 2.0;
          //RavlInfo("Updating thresh %0.4f", thresh);
          //thresh = it.Data().Data2();

        }

        // update the prev score
        prevScore = currentScore;
      }

      ONDEBUG(cout << "Threshold: " << thresh << " EER: " << eer << endl);
      return thresh;
    }
    
    RealT RocBodyC::SmallestErrorRate() const
    {
      //: Check it is sorted before use
      if (!sorted)
        return -1.0;

      if (!IsValid())
        return -1.0;

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;
      RealT minSoFar = 1.0e10;
      RealT thresh = 0.0;

      for (ConstDLIterC<ClaimT> it(claims); it; it++) {

        if (it.Data().Data1())
          clientssofar += 1.0;
        else
          impostorssofar += 1.0;

        RealT clientsrejected = clients - clientssofar;
        RealT impostorsaccepted = impostorssofar;

        //: the clever bit i hope
        RealT fa, fr;
        if (impostors > 0)
          fa = impostorsaccepted / impostors;
        else
          fa = 0.0;
        if (clients > 0)
          fr = clientsrejected / clients;
        else
          fr = 0.0;

        RealT dif = (fr + fa) / 2.0;
        if (dif < minSoFar) {
          minSoFar = dif;
          thresh = it.Data().Data2();
        }
      }ONDEBUG(cout << "Threshold: " << thresh << " HTER: " << minSoFar << endl);
      return thresh;
    }
    
    RealT RocBodyC::FAZero() const
    {
      //: Check it is sorted before use
      if (!sorted)
        return -1.0;

      if (!IsValid())
        return -1.0;

      RealT clientssofar = 0.0;
      RealT thresh = 0.0;

      ConstDLIterC<ClaimT> it(claims);
      bool done = false;
      while (!done) {

        if (it.Data().Data1())
          clientssofar += 1.0;
        else {
          thresh = it.Data().Data2();
          done = true;
        }

        it.Next();
        if (!it.IsElm())
          done = true;
      }

      ONDEBUG(cout << "FAZero: " << (1.0- (clientssofar/clients)) << endl);
      return thresh;
    }

    RealT RocBodyC::ErrorRateFA(RealT desError) const
    {
      //: Check it is sorted before use
      if (!sorted)
        return -1.0;

      if (!IsValid())
        return -1.0;

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;
      RealT thresh = 0.0;

      ConstDLIterC<ClaimT> it(claims);
      bool done = false;
      while (!done) {

        if (it.Data().Data1())
          clientssofar += 1.0;
        else {
          impostorssofar += 1.0;
          RealT error = impostorssofar / impostors;
          if (error > desError) {
            thresh = it.Data().Data2();
            ONDEBUG(cout << "Threshold: " << thresh << endl);
            done = true;
          }
        }

        it.Next();
        if (!it.IsElm())
          done = true;
      }

      ONDEBUG(cout << "FA Error Rate: " << impostorssofar/impostors << endl);ONDEBUG(cout << "FR Error Rate: " << (1.0 - (clientssofar/clients)) << endl);
      return thresh;
    }

    RealT RocBodyC::ErrorRateFR(RealT desError) const
    {
      //: Check it is sorted before use
      if (!sorted)
        return -1.0;

      if (!IsValid())
        return -1.0;

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;
      RealT thresh = 0.0;

      ConstDLIterC<ClaimT> it(claims);
      bool done = false;
      while (!done) {

        //cout << RkIt.Data().Data1() << " " << RkIt.Data().Data2() << endl;
        if (it.Data().Data1()) {
          clientssofar += 1.0;
          RealT error = 1.0 - (clientssofar / clients);
          //cout << "Error: " << error << endl;
          if (error < desError) {
            thresh = it.Data().Data2();
            cout << "Threshold: " << thresh << endl;
            done = true;
          }
        } else {
          impostorssofar += 1.0;
        }

        it.Next();
        if (!it.IsElm())
          done = true;
      }

      ONDEBUG(cout << "FA Error Rate: " << impostorssofar/impostors << endl);ONDEBUG(cout << "FR Error Rate: " << (1.0 - (clientssofar/clients)) << endl);
      return thresh;
    }
    
    ResultsInfoC RocBodyC::Error(RealT threshold) const
    {
      //: Check it is sorted before use
      if (!sorted)
        return ResultsInfoC();

      if (!IsValid())
        return ResultsInfoC();

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;

      ConstDLIterC<ClaimT> it(claims);
      bool doing = true;
      while (doing) {
        //:  Impostor model
        if (highMatches) {
          if (it.Data().Data2() > threshold)
            doing = false;
          else {
            if (it.Data().Data1())
              clientssofar += 1.0;
            else
              impostorssofar += 1.0;
          }
        }
        //: Client model
        else {
          if (it.Data().Data2() < threshold)
            doing = false;
          else {
            if (it.Data().Data1())
              clientssofar += 1.0;
            else
              impostorssofar += 1.0;
          }
        }
        it++;
        doing = it.IsElm();
      }
      //: Lets work out FA and FR rates
      RealT fa, fr;
      if (highMatches) {
        fa = (impostors - impostorssofar) / impostors;
        fr = clientssofar / clients;
      } else {
        fa = impostorssofar / impostors;
        fr = (clients - clientssofar) / clients;
      }

      ResultsInfoC results(fa, fr, threshold);

      ONDEBUG(cout << "fa: " << fa << " fr: " << fr << endl);

      return results;
    }
    
    RealHistogram1dC RocBodyC::Histogram(bool forClients) const
    {
      //: Check it is sorted before use
      if (!sorted)
        return RealHistogram1dC();

      if (!IsValid())
        return RealHistogram1dC();

      //: Are we interested in client or impostor distribution
      ConstDLIterC<ClaimT> it(claims);
      DListC<RealT> values;
      for (it.First(); it; it++) {
        if (forClients && it.Data().Data1())
          values.InsLast(it.Data().Data2());
        else if (!forClients && !it.Data().Data1())
          values.InsLast(it.Data().Data2());
      }

      //:Find the min and maximum of whole data
      it.First();
      RealT min = it.Data().Data2();
      RealT max = it.Data().Data2();
      for (it++; it; it++) {
        if (it.Data().Data2() < min)
          min = it.Data().Data2();
        if (it.Data().Data2() > max)
          max = it.Data().Data2();
      }

      //: Finally write the histogram
      RealHistogram1dC hist(min, max, 100);
      for (ConstDLIterC<RealT> it2(values); it2; it2++) {
        hist.Vote(it2.Data());
      }
      return hist;
    }
    
    DListC<Tuple2C<RealT, RealT> > RocBodyC::Graph(RealT maxFa, RealT maxFr) const
    {

      //: Check it is sorted before use
      if (!sorted)
        return DListC<Tuple2C<RealT, RealT> >();

      if (!IsValid())
        return DListC<Tuple2C<RealT, RealT> >();

      RealT clientssofar = 0.0;
      RealT impostorssofar = 0.0;
      DListC<Tuple2C<RealT, RealT> > data;

      for (ConstDLIterC<ClaimT> it(claims); it; it++) {

        if (it.Data().Data1())
          clientssofar += 1.0;
        else
          impostorssofar += 1.0;

        RealT clientsrejected = clients - clientssofar;
        RealT impostorsaccepted = impostorssofar;

        //: the clever bit i hope
        RealT fa, fr;
        if (impostors > 0)
          fa = impostorsaccepted / impostors;
        else
          fa = 0.0;
        if (clients > 0)
          fr = clientsrejected / clients;
        else
          fr = 0.0;
        //: update date structure
        Tuple2C<RealT, RealT> elem(fa, fr);
        if (fa < maxFa && fr < maxFr)
          data.InsLast(elem);
      }
      return data;
    }
    
    SArray1dC<Tuple3C<RealT, RealT, RealT> > RocBodyC::ThresholdInfo() const
    {

      //: Check it is sorted before use
      if (!sorted)
        return SArray1dC<Tuple3C<RealT, RealT, RealT> >();

      //: this generates information about the
      //: Roc curve that get put into the ABI model
      if (!IsValid())
        return SArray1dC<Tuple3C<RealT, RealT, RealT> >();

      SArray1dC<Tuple3C<RealT, RealT, RealT> > data(101);
      //: OK we are can vary the threshold between 0 and 100
      //: and work out the relevant fa and fr rates
      //: not the most effecient but small overhead
      //: compared to everything else we are doing
      for (UIntT thresh = 0; thresh < 101; thresh++) {
        ResultsInfoC res = Error((RealT) thresh);
        data[thresh] = Tuple3C<RealT, RealT, RealT>(res.FA() * 100.0, res.FR() * 100.0, res.Threshold());
      }
      return data;
    }
    
    GnuPlotC RocBodyC::Plot(RealT maxFa, RealT maxFr) const
    {

      //: Check it is sorted before use
      if (!sorted)
        return GnuPlotC();

      if (!IsValid())
        return GnuPlotC();

      DListC<Tuple2C<RealT, RealT> > points = Graph(maxFa, maxFr);
      FilenameC nn = "/tmp/graph.png";
      GnuPlotC plot;
      for (DLIterC<Tuple2C<RealT, RealT> > it(points); it; it++) {
        plot.AddPoint(0, it.Data().Data1(), it.Data().Data2());
      }
      plot.Terminal("png");
      plot.Title("ROC Curve");
      plot.Xlabel("FA");
      plot.Ylabel("FR");
      plot.Xlo(0.0);
      plot.Xhi(maxFa);
      plot.Ylo(0.0);
      plot.Yhi(maxFr);
      return plot;
    }

    GnuPlotC RocBodyC::PlotHistogram(UIntT type) const
    {
      //: Check it is sorted before use
      if (!sorted)
        return GnuPlotC();

      if (!IsValid())
        return GnuPlotC();

      // Initialise client or impostor histogram
      RealHistogram1dC clientHist = Histogram(true);
      RealHistogram1dC impostHist = Histogram(false);

      // Assign histogram points
      FilenameC fileName = "/tmp/histogram.png";
      GnuPlotC plot;
      UIntT maxY = 0;
      if (type == 0 || type == 2) {
        for (SArray1dIterC<UIntC> it(clientHist); it; it++) {
          plot.AddPoint(0, clientHist.MidBin(it.Index()), it.Data());
          if (it.Data() > maxY)
            maxY = it.Data();
        }
      } else if (type == 1 || type == 2) {
        for (SArray1dIterC<UIntC> it(impostHist); it; it++) {
          plot.AddPoint(0, impostHist.MidBin(it.Index()), it.Data());
          if (it.Data() > maxY)
            maxY = it.Data();
        }
      }

      // Assign histogram characteristics
      plot.Terminal("png");
      plot.Title("Histogram");
      plot.Xlabel("Score");
      plot.Ylabel("Number of claims");
      plot.Xlo(impostHist.MinLimit());
      plot.Xhi(clientHist.MaxLimit());
      plot.Ylo(0);
      plot.Yhi(maxY);

      return plot;
    }

    /*
     * Get the maximum score in the ROC
     */
    RealT RocBodyC::MaxFAScore() const {
      if(!sorted) {
        return -1.0;
      }
      for(DLIterC<ClaimT>it(claims);it;it++) {
        if(it.Data().Data1())
          continue;
        return it.Data().Data2();
      }
      return -1.0;
    }

    /*
     * Get the maximum score in the ROC
     */
    RealT RocBodyC::MinFRScore() const
    {
      if (!sorted) {
        return -1.0;
      }

      DListC<ClaimT>r = claims.Copy();
      r.Reverse();
      for(DLIterC<ClaimT>it(r);it;it++) {
        if(!it.Data().Data2())
          continue;
        return it.Data().Data2();
      }
      return -1.0;
    }


    bool RocBodyC::IsValid() const
    {
      if (clients == 0 || impostors == 0)
        return false;
      return true;
    }
    
    // Lets report some numbers at some key-points
    bool RocBodyC::Report(const DirectoryC & outDir)
    {
      Sort();
      SArray1dC<RealT> fa(3);
      StrIStreamC("3 0.01 0.001 0.0001") >> fa;
      RCHashC<RealT, RealT> fa2fr;
      StringC summary;
      OStreamC os(outDir + "/results.txt");
      RealT eerThreshold = EqualErrorRate();
      ResultsInfoC resInfo = Error(0.0);
      summary.form("Equal Error Rate FA=%0.4f%%, FR=%0.4f%% with Threshold=%0.4f\n",
          resInfo.FA() * 100.0,
          resInfo.FR() * 100.0,
          eerThreshold);
      for (SArray1dIterC<RealT> it(fa); it; it++) {
        RealT threshold = ErrorRateFA(*it);
        ResultsInfoC resInfo = Error(threshold);
        StringC res;
        res.form("At FA=%0.4f%%, Threshold=%0.4f and FR=%0.4f%%", *it * 100.0, threshold, resInfo.FR() * 100.0);
        RavlDebug("%s", res.data());
        os << res << endl;
        fa2fr.Insert(*it, resInfo.FR());
        StringC res2;
        res2.form("(%0.4f%%, %0.4f%%) ", *it * 100.0, resInfo.FR() * 100.0);
        summary += res2;
      }
      RavlInfo("%s", summary.data());

      // Lets plot the ROC
      RealT rng = fa2fr[0.1] * 1.2;
      if (rng > 1.0 || rng <= 0.0) {
        rng = 1.0;
      }

      RavlGUIN::GnuPlotC plot = Plot(rng, rng);
      plot.SetAxisRange(0, rng, 0, rng);
      plot.Plot(outDir + "/roc.png");

      // Lets generate some score distributions
      RavlGUIN::GnuPlotC distPlot0 = PlotHistogram(0);
      distPlot0.Plot(outDir + "/clientScoreDistribution.png");

      RavlGUIN::GnuPlotC distPlot1 = PlotHistogram(1);
      distPlot1.Plot(outDir + "/impostorScoreDistribution.png");

      return true;
    }

// output your class members
    ostream &
    operator<<(ostream & s, const RocBodyC & out)
    {
      //: Check it is sorted before use
      s << out.claims << endl;
      s << out.metricType << endl;
      s << out.highMatches << endl;
      s << out.clients << endl;
      s << out.impostors;
      return s;
    }
    
    BinOStreamC &
    operator<<(BinOStreamC & s, const RocBodyC & out)
    {
      //: Check it is sorted before use
      s << out.claims;
      s << out.metricType;
      s << out.highMatches;
      s << out.clients;
      s << out.impostors;
      return s;
    }
    
//////////////////////////////
// input your class members
    
    istream &
    operator>>(istream & s, RocBodyC & in)
    {
      in = RocBodyC(s);
      return s;
    }

    void InitRocIO()
    {
    }
    // RAVL I/O

    FileFormatStreamC<RocC> FileFormatStream_RocC;
    FileFormatBinStreamC<RocC> FileFormatBinStream_RocC;
    static TypeNameC typenameRoc(typeid(RocC), "RocC");

  }
} // end namespace