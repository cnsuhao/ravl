// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/Genetic/EvaluateFitness.hh"

namespace RavlN {  namespace GeneticN {

  //! Constructor
  EvaluateFitnessC::EvaluateFitnessC()
  {}

  //! Factory constructor
  EvaluateFitnessC::EvaluateFitnessC(const XMLFactoryContextC &factory)
  {}

  //! Copy me.
  RCBodyVC &EvaluateFitnessC::Copy() const
  { return *new EvaluateFitnessC(*this); }

  //! Access type of object evaluated for fitness.
  const type_info &EvaluateFitnessC::ObjectType() const
  { return typeid(void); }

  //! Generate a new problem in the domain.
  bool EvaluateFitnessC::GenerateNewProblem() {
    return true;
  }

  //! Evaluate the fit
  bool EvaluateFitnessC::Evaluate(RCWrapAbstractC &obj,float &score)
  {
    RavlAssertMsg(0,"Abstract method called.");
    return false;
  }

  //: Called when owner handles drop to zero.
  void EvaluateFitnessC::ZeroOwners() {
    RCLayerBodyC::ZeroOwners();
  }

 }}
