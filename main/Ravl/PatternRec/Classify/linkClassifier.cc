

namespace RavlN {

  extern void linkDesignClassifierSupervised();
  extern void linkDesignKNearestNeighbour();
  extern void linkDesignClassifierGaussianMixture();

  void LinkClassifier() {
    linkDesignClassifierSupervised();
    linkDesignKNearestNeighbour();
    linkDesignClassifierGaussianMixture();
  }

}
