
#include "Ravl/PatternRec/DesignClassifierNeuralNetwork2.hh"
#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/Genetic/GenomeConst.hh"


namespace RavlN {

  DesignClassifierNeuralNetwork2C GeneFactory2NN2(const RavlN::GeneticN::GeneFactoryC &factory)
  {
    static RavlN::GeneticN::GeneTypeIntC::RefT numLayersType = new RavlN::GeneticN::GeneTypeIntC("layers",2,6);
    static RavlN::GeneticN::GeneTypeIntC::RefT numHiddenType = new RavlN::GeneticN::GeneTypeIntC("hidden",2,200);
    static RavlN::GeneticN::GeneTypeFloatC::RefT numReg = new RavlN::GeneticN::GeneTypeFloatC("regularisation",0.0,150);

#if 0
    UIntT nfeatures;
    if(!factory.GenePalette().GetParameter("features",nfeatures))
      nfeatures = 100;
#endif

    RealT regularisation = 0;

    IntT nLayers = 0;
    IntT nHidden = 0;

    factory.Get("Layers",nLayers,*numLayersType);
    factory.Get("Hidden",nHidden,*numHiddenType);
    factory.Get("Regularisation",regularisation,*numReg);

    return DesignClassifierNeuralNetwork2C(nLayers,nHidden,false,regularisation,0.0001,500);
  }

  DP_REGISTER_CONVERSION(GeneFactory2NN2,1.0);

  void LinkGeneticClassifierConstructors()
  {}

}
