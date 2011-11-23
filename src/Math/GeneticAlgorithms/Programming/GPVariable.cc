// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticProgram
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Programming

#include "Ravl/Genetic/GPVariable.hh"
#include "Ravl/Genetic/GenomeClass.hh"
#include "Ravl/Threads/Signal1.hh"
#include "Ravl/Calls.hh"
#include "Ravl/Array1d.hh"

namespace RavlN { namespace GeneticN {

  //! Factory constructor
  GPVariableBaseC::GPVariableBaseC(const XMLFactoryContextC &factory)
   : m_isSet(false)
  {

  }

  //! Factory constructor
  GPVariableBaseC::GPVariableBaseC(const GeneFactoryC &factory)
  : m_isSet(false)
  {

  }

  //! Constructor
  GPVariableBaseC::GPVariableBaseC()
   : m_isSet(false)
  {

  }

  static RegisterGeneClassC<GPVariableC<IntT> > g_registerGPVariableInt("RavlN::GeneticN::GPVariableC<IntT>");
  static RegisterGeneClassC<GPVariableC<float> > g_registerGPVariableFloat("RavlN::GeneticN::GPVariableC<float>");

  void LinkGPVariable()
  {}
}}
