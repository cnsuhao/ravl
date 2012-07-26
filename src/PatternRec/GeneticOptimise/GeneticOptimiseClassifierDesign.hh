#ifndef RAVL_GENETICOPTIMISECLASSIFIERDESIGN_HEADER
#define RAVL_GENETICOPTIMISECLASSIFIERDESIGN_HEADER 1

#include "Ravl/PatternRec/DesignClassifierSupervised.hh"
#include "Ravl/PatternRec/Error.hh"
#include "Ravl/Genetic/GeneticOptimiser.hh"

namespace RavlN {

  //: Optimise classifier parameters for generalisation

  class GeneticOptimiseClassifierDesignBodyC
   : public DesignFunctionSupervisedBodyC
  {
  public:
    //! Constructor
    GeneticOptimiseClassifierDesignBodyC();

    //! Constructor
    GeneticOptimiseClassifierDesignBodyC(RealT designFraction);

    //! XML factory constructor
    GeneticOptimiseClassifierDesignBodyC(const XMLFactoryContextC &factory);

    //! Optimise a the design for a classifier given some data and a designer
    virtual bool Apply(const SampleVectorC & x,
                       const SampleLabelC & labels,
                       ClassifierC &bestClassifier,
                       RealT &finalResult
                       ) ;

    typedef RavlN::SmartPtrC<GeneticOptimiseClassifierDesignBodyC> RefT;
  protected:
    GeneticN::GeneticOptimiserC::RefT m_optimiser;
    ErrorC m_errorMeasure;  //!<
    RealT m_designFraction; // Fraction of the training set used for design.
    RealT m_crossValidateFraction; // Fraction used for cross validation, the remaining fraction is used for testing.
  };

  //: Optimise classifier parameters for generalisation

  class GeneticOptimiseClassifierDesignC
  : public DesignFunctionSupervisedC
  {
  public:
    GeneticOptimiseClassifierDesignC()
    {}

    //: Create a designer
    GeneticOptimiseClassifierDesignC(RealT designFraction)
      : DesignFunctionSupervisedC(new GeneticOptimiseClassifierDesignBodyC(designFraction))
    {}

    //: Create a designer
    GeneticOptimiseClassifierDesignC(const XMLFactoryContextC &factory)
      : DesignFunctionSupervisedC(new GeneticOptimiseClassifierDesignBodyC(factory))
    {}

  protected:
    GeneticOptimiseClassifierDesignBodyC &Body()
    { return static_cast<GeneticOptimiseClassifierDesignBodyC &>(DesignerC::Body()); }
    //: Access body.

    const GeneticOptimiseClassifierDesignBodyC &Body() const
    { return static_cast<const GeneticOptimiseClassifierDesignBodyC &>(DesignerC::Body()); }
    //: Access body.

  public:

    bool Apply(        const SampleVectorC & sample,
                       const SampleLabelC & labels,
                       ClassifierC &bestClassifier,
                       RealT &finalResult
                       )
    { return Body().Apply(sample,labels,bestClassifier,finalResult); }

  };

}

#endif
