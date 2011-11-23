// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENOMECONST_HH
#define RAVL_GENETIC_GENOMECONST_HH 1
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/Genetic/Genome.hh"

namespace RavlN { namespace GeneticN {

  //! Type informat for int
  class GeneTypeIntC
   : public GeneTypeC
  {
  public:
    //! Factory constructor
    GeneTypeIntC(const XMLFactoryContextC &factory);

    //! Constructor
    GeneTypeIntC(const std::string &name,IntT min,IntT max);

    //! Load form a binary stream
    GeneTypeIntC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneTypeIntC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Create randomise value
    virtual void Random(GeneC::RefT &newValue) const;

    //! Mutate a gene
    virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const;

    //! Access min value
    IntT Min() const
    { return m_min; }

    //! Access max value
    IntT Max() const
    { return m_max; }

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneTypeIntC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneTypeIntC> ConstRefT;
  protected:
    IntT m_min;
    IntT m_max;
  };

  //! Gene for integer variable
  class GeneIntC
   : public GeneC
  {
  public:
    //! Factory constructor
    GeneIntC(const XMLFactoryContextC &factory);

    //! Constructor
    GeneIntC(const GeneTypeC &geneType,IntT value);

    //! Load form a binary stream
    GeneIntC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneIntC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Save to binary stream
    virtual bool Save(RavlN::XMLOStreamC &strm) const;

    //! Access value
    IntT Value() const
    { return m_value; }

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneIntC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneIntC> ConstRefT;
  protected:
    IntT m_value;
  };

     //! Type information for float
   class GeneTypeFloatC
    : public GeneTypeC
   {
   public:
     //! Factory constructor
     GeneTypeFloatC(const XMLFactoryContextC &factory);

     //! Constructor
     GeneTypeFloatC(const std::string &name,float min,float max);

     //! Load form a binary stream
     GeneTypeFloatC(BinIStreamC &strm);

     //! Load form a binary stream
     GeneTypeFloatC(std::istream &strm);

     //! Save to binary stream
     virtual bool Save(BinOStreamC &strm) const;

     //! Save to binary stream
     virtual bool Save(std::ostream &strm) const;

     //! Generate a new value
     virtual void RandomValue(float &value) const;

     //! Create randomise value
     virtual void Random(GeneC::RefT &newValue) const;

     //! Mutate a gene
     virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const;

     //! Access min value
     float Min() const
     { return m_min; }

     //! Access max value
     float Max() const
     { return m_max; }

     // Reference to this gene.
     typedef RavlN::SmartPtrC<GeneTypeFloatC> RefT;

     // Const reference to this gene.
     typedef RavlN::SmartPtrC<const GeneTypeFloatC> ConstRefT;
   protected:
     float m_min;
     float m_max;
   };

  //! Gene for floating point variable

  class GeneFloatC
   : public GeneC
  {
  public:
     //! Factory constructor
     GeneFloatC(const XMLFactoryContextC &factory);

     //! Constructor
     GeneFloatC(const GeneTypeC &geneType,float value);

     //! Load form a binary stream
     GeneFloatC(BinIStreamC &strm);

     //! Load form a binary stream
     GeneFloatC(std::istream &strm);

     //! Save to binary stream
     virtual bool Save(BinOStreamC &strm) const;

     //! Save to binary stream
     virtual bool Save(std::ostream &strm) const;

     //! Save to binary stream
     virtual bool Save(RavlN::XMLOStreamC &strm) const;

     //! Access value
     float Value() const
     { return m_value; }

     // Reference to this gene.
     typedef RavlN::SmartPtrC<GeneFloatC> RefT;

     // Const reference to this gene.
     typedef RavlN::SmartPtrC<const GeneFloatC> ConstRefT;
   protected:
      float m_value;
   };

}}

#endif
