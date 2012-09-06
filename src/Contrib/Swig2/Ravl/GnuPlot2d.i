// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/String.i"
%include "Ravl/Swig2/Point2d.i"
%include "Ravl/Swig2/Index.i"
%include "Ravl/Swig2/RealRange.i"
%include "Ravl/Swig2/IndexRange2d.i"
%include "Ravl/Swig2/SArray1d.i"
%include "Ravl/Swig2/Hash.i"
%include "Ravl/Swig2/Collection.i"

%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/Plot/GnuPlot2d.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  class GnuPlot2dC {
  public:
    GnuPlot2dC(const StringC & title = "My Plot");
    //: Construct with a title

    bool Plot(const SArray1dC<Point2dC> & points, const StringC & dataName = "");
    //: Plot all graphs on single canvas

    bool Plot(const RCHashC<StringC, CollectionC<Point2dC> > & data);
    //: Plot all points on same graph

    bool Plot(const StringC & function);
    //: Plot a function
 
 	//bool Plot(const LineABC2dC & line);
 
    bool Plot(const DataSetVectorLabelC & dataSet, UIntT fv1 = 0, UIntT fv2 = 1, UIntT samplesPerClass = 0);
    //: Make a scatter plot of the data.  Only the first two dimensions will be used...

	bool Plot(const ClassifierC & classifier, const DataSetVectorLabelC & dataSet, UIntT feature1=0, UIntT feature2=1);
	//: Plot a classifier decision boundary...works best for 2D classifiers
	
    bool SetXLabel(const StringC & xlabel);
    //: Set the x-label

    bool SetYLabel(const StringC & ylabel);
    //: Set the y-label

    bool SetXRange(const RealRangeC & xrange);
    //: Set the range of the x-axis

    bool SetYRange(const RealRangeC & yrange);
    //: Set the range of the y-axis

    bool SetLineStyle(const StringC & lineStyle);
    //: Set plot style

    bool SetOutput(const StringC & output, const IndexRange2dC & rec = IndexRange2dC(512, 512));
    //: Set output x11 or a filename

    bool Command(const StringC & command);
    //: General method to send a command to the plotting library

  };
	
  %pythoncode %{
  	def Plot(data, title = "My Data"):
  		"""
  		Try to automatically plot anything then GnuPlot2d is capable of
  		"""
  		gnuplot = GnuPlot2dC(title)
  		gnuplot.Plot(data)
  		return gnuplot 
  		
  	def PlotFile(data, filename, title = "My Data", rows=750, cols=1000):
  		"""
  		Try to automatically plot anything as an image and save it to file that GnuPlot2d is capable of.
  		"""
  		gnuplot = GnuPlot2dC(title)
  		rec = IndexRange2dC(rows, cols);
  		gnuplot.SetOutput(filename, rec)
  		gnuplot.Plot(data)
  		return gnuplot
  %}

}
