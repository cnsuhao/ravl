// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2005-12, University of Surrey.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlPatternRec


namespace RavlN {

  extern void linkDesignClassifierSupervised();
  extern void linkDesignKNearestNeighbour();
  extern void linkDesignClassifierGaussianMixture();
  extern void linkDesignOneAgainstAll();
  extern void InitRavlClassifierPreprocess();
  extern void InitRavlClassifierNeuralNetwork();
  extern void InitRavlClassifierLinearCombinationIO();
  extern void linkDesignClassifierNeuralNetwork();
  extern void linkDesignClassifierLogisticRegression();

  void LinkClassifier() {
    linkDesignClassifierSupervised();
    linkDesignKNearestNeighbour();
    linkDesignClassifierGaussianMixture();
    linkDesignOneAgainstAll();
    InitRavlClassifierPreprocess();
    InitRavlClassifierNeuralNetwork();
    InitRavlClassifierLinearCombinationIO();
    linkDesignClassifierNeuralNetwork();
    linkDesignClassifierLogisticRegression();
  }

}
