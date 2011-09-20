

namespace RavlN {

  extern void linkDesignClassifierSupervised();
  extern void linkDesignKNearestNeighbour();
  extern void linkDesignClassifierGaussianMixture();
  extern void linkDesignOneAgainstAll();
  extern void InitRavlClassifierPreprocess();
  extern void InitRavlClassifierNeuralNetwork();
  extern void InitRavlClassifierLinearCombinationIO();
  extern void linkDesignClassifierNeuralNetwork();

  void LinkClassifier() {
    linkDesignClassifierSupervised();
    linkDesignKNearestNeighbour();
    linkDesignClassifierGaussianMixture();
    linkDesignOneAgainstAll();
    InitRavlClassifierPreprocess();
    InitRavlClassifierNeuralNetwork();
    InitRavlClassifierLinearCombinationIO();
    linkDesignClassifierNeuralNetwork();
  }

}
