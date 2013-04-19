/*
 * GeneticOptimiseClassifierDesign.cc
 *
 *  Created on: 25 Jun 2012
 *      Author: charles
 */

#include "Ravl/PatternRec/GeneticOptimiseClassifierDesign.hh"
#include "Ravl/PatternRec/ClassifierPreprocess.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/PatternRec/EvaluateGeneticClassifier.hh"
#include "Ravl/Genetic/GeneFactory.hh"

#define DODEBUG	0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //! Constructor
  GeneticOptimiseClassifierDesignBodyC::GeneticOptimiseClassifierDesignBodyC()
   : m_designFraction(0.6),
     m_crossValidateFraction(0.2)
  {}

  //! Constructor
  GeneticOptimiseClassifierDesignBodyC::GeneticOptimiseClassifierDesignBodyC(RealT designFraction)
   : m_designFraction(designFraction),
     m_crossValidateFraction(0.2)
  {}

  //! XML factory constructor
  GeneticOptimiseClassifierDesignBodyC::GeneticOptimiseClassifierDesignBodyC(const XMLFactoryContextC &factory)
   : m_designFraction(factory.AttributeReal("designFraction",0.6)),
     m_crossValidateFraction(factory.AttributeReal("crossValidationFraction",0.2))
  {
    factory.UseChildComponent("ErrorMeasure",m_errorMeasure,true,typeid(ErrorC));
    factory.UseChildComponent("Optimiser",m_optimiser,false,typeid(GeneticN::GeneticOptimiserC));
  }

  //! Optimise a the design for a classifier given some data and a designer
  bool GeneticOptimiseClassifierDesignBodyC::Apply(const SampleVectorC & sample,
                                             const SampleLabelC & labels,
                                             ClassifierC &bestClassifier,
                                             RealT &finalResult
                                             )
  {
    DataSetVectorLabelC trainSample(sample,labels);
    DataSetVectorLabelC crossValidateSample(trainSample.ExtractSample(1 - m_designFraction));

    // We only need to do normalisation once, so get that out of the way.

    MeanCovarianceC meanCovariance = trainSample.Sample1().MeanCovariance();
    FunctionC normaliseFunc = trainSample.Sample1().NormalisationFunction(meanCovariance);

    trainSample.Sample1().Normalise(meanCovariance);
    crossValidateSample.Sample1().Normalise(meanCovariance);

    EvaluateGeneticClassiferC::RefT eval = new EvaluateGeneticClassiferC(m_errorMeasure,
                                                                        trainSample,
                                                                        crossValidateSample);
    m_optimiser->SetFitnessFunction(*eval);

    // Run the optimisation
    m_optimiser->Run();

    // Get the genome for the best solution.
    GeneticN::GenomeC::RefT genome;
    float score = 0;
    if(!m_optimiser->GetBestGenome(genome,score))
      return false;

    finalResult = score;

    // Create designer matching the best specification.
    GeneticN::GeneFactoryC factory(*genome,m_optimiser->GenePalette());
    DesignClassifierSupervisedC designer;
    factory.Get(designer);

    // Retrain on all the data.
    ClassifierC classifier = designer.Apply(sample,labels);

    // Add normalisation in to make the final classifier.
    bestClassifier = ClassifierPreprocessC(normaliseFunc, classifier);

    return true;
  }

  void LinkGeneticOptimiseClassifierDesign()
  {}

  static RavlN::XMLFactoryRegisterHandleConvertC<GeneticOptimiseClassifierDesignC,DesignFunctionSupervisedC>
      g_registerXMLFactoryGeneticOptimiseClassifierDesign("RavlN::GeneticOptimiseClassifierDesignC");

}

