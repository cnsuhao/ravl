// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! author=Charles Galambos
//! lib=RavlGeneticOptimisation
//! docentry=Ravl.API.Math.Genetic.Optimisation

#ifndef RAVL_EVALUATEGENETICCLASSIFER_HH_
#define RAVL_EVALUATEGENETICCLASSIFER_HH_

#include "Ravl/Genetic/EvaluateFitness.hh"
#include "Ravl/PatternRec/DesignClassifierSupervised.hh"
#include "Ravl/PatternRec/Error.hh"

namespace RavlN {

  //! Abstract class for evaluating the fitness of an object.

  class EvaluateGeneticClassiferC
   : public GeneticN::EvaluateFitnessC
  {
  public:
    //! Constructor
    EvaluateGeneticClassiferC();

    //! Constructor
    EvaluateGeneticClassiferC(const ErrorC &errorMeasure,
                              const DataSetVectorLabelC &trainSample,
                              const DataSetVectorLabelC &crossValidateSample);

    //! Copy constructor
    EvaluateGeneticClassiferC(const EvaluateGeneticClassiferC &other);

    //! Copy me.
    virtual RavlN::RCBodyVC &Copy() const;

    //! Access type of object evaluated for fitness.
    virtual const std::type_info &ObjectType() const;

    //! Evaluate the fit
    virtual bool Evaluate(RCWrapAbstractC &obj,float &score);

    //! Handle to this class.
    typedef RavlN::SmartOwnerPtrC<EvaluateGeneticClassiferC> RefT;

    //! Handle to class
    typedef RavlN::SmartCallbackPtrC<EvaluateGeneticClassiferC> CBRefT;

  protected:
    virtual void ZeroOwners()
    { EvaluateFitnessC::ZeroOwners(); }
    //: Called when owner handles drop to zero.

    mutable DesignClassifierSupervisedC m_designer;
    mutable ErrorC m_errorMeasure;  //!<
    DataSetVectorLabelC m_trainSample;
    DataSetVectorLabelC m_crossValidateSample;
  };
}

#endif /* EVALUATOR_HH_ */
