// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/*
 * Evaluator.hh
 *
 *  Created on: Sep 24, 2011
 *      Author: charlesgalambos
 */
//! docentry=Ravl.API.Math.Genetic.Optimisation
//! author=Charles Galambos
//! lib=RavlGeneticOptimisation

#ifndef RAVL_GENETIC_EVALUATEFITNESS_HH_
#define RAVL_GENETIC_EVALUATEFITNESS_HH_

#include "Ravl/SmartLayerPtr.hh"
#include "Ravl/RCWrap.hh"

namespace RavlN {
  class XMLFactoryContextC;
}

namespace RavlN {  namespace GeneticN {
  using RavlN::XMLFactoryContextC;

  //! Abstract class for evaluating the fitness of an object.

  class EvaluateFitnessC
   : public RavlN::RCLayerBodyC
  {
  public:
    //! Constructor
    EvaluateFitnessC();

    //! Factory constructor
    EvaluateFitnessC(const XMLFactoryContextC &factory);

    //! Copy me.
    virtual RCBodyVC &Copy() const = 0;

    //! Access type of object evaluated for fitness.
    virtual const std::type_info &ObjectType() const = 0;

    //! Generate a new problem in the domain.
    virtual bool GenerateNewProblem();

    //! Evaluate the fit
    virtual bool Evaluate(RCWrapAbstractC &obj,float &score) = 0;

    //! Handle to this class.
    typedef RavlN::SmartOwnerPtrC<EvaluateFitnessC> RefT;

    //! Handle to class
    typedef RavlN::SmartCallbackPtrC<EvaluateFitnessC> CBRefT;

  protected:
    virtual void ZeroOwners();
    //: Called when owner handles drop to zero.

  };
}}

#endif /* EVALUATOR_HH_ */
