// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_GENETIC_GENOMEList_HH
#define RAVL_GENETIC_GENOMEList_HH 1
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/Genetic/Genome.hh"
#include "Ravl/Collection.hh"
#include "Ravl/DP/TypeConverter.hh"

namespace RavlN { namespace GeneticN {

  //! Base class for Lists with child fields.
  class GeneTypeListBaseC
   : public GeneTypeC
  {
  public:
    //! Factory constructor
    GeneTypeListBaseC(const XMLFactoryContextC &factory);

    // Constructor
    GeneTypeListBaseC(const std::string &name,const GeneTypeC &contentType,UIntT maxSize);

    //! Load form a binary stream
    GeneTypeListBaseC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneTypeListBaseC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Create randomise value
    virtual void Random(GeneC::RefT &newValue) const;

    //! Mutate a gene
    virtual bool Mutate(float fraction,const GeneC &original,RavlN::SmartPtrC<GeneC> &newValue) const;

    //! Mutate a gene
    virtual void Cross(const GeneC &original1,const GeneC &original2,RavlN::SmartPtrC<GeneC> &newValue) const;

    //! Create a new list
    virtual bool CreateList(RCWrapAbstractC &list) const;

    //! Add item to list.
    virtual bool AddToList(RCWrapAbstractC &list,RCWrapAbstractC &item) const;

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneTypeListBaseC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneTypeListBaseC> ConstRefT;

  protected:
    GeneTypeC::ConstRefT m_contentType;
    UIntT m_maxLength;
  };

  template<typename EntryT>
  class GeneTypeListC
   : public GeneTypeListBaseC
  {
  public:
    //! Factory constructor
    GeneTypeListC(const XMLFactoryContextC &factory)
     : GeneTypeListBaseC(factory)
    {}

    // Constructor
    GeneTypeListC(const std::string &name,const GeneTypeC &contentType,UIntT maxSize)
     : GeneTypeListBaseC(name,contentType,maxSize)
    {}

    //! Load form a binary stream
    GeneTypeListC(BinIStreamC &strm)
     : GeneTypeListBaseC(strm)
    {
      ByteT version = 0;
      strm >> version;
      if(version != 1)
        throw RavlN::ExceptionUnexpectedVersionInStreamC("GeneTypeListC");
    }

     //! Load form a binary stream
    GeneTypeListC(std::istream &strm)
      : GeneTypeListBaseC(strm)
    {}

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const
    {
      GeneTypeListBaseC::Save(strm);
      ByteT version = 1;
      strm << version;
      return true;
    }

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const
    {
      GeneTypeListBaseC::Save(strm);
      return false;
    }

    //! Create a new list
    virtual bool CreateList(RCWrapAbstractC &list) const {
      CollectionC<EntryT> newCollection(32);
      list = RCWrapC<CollectionC<EntryT> >(newCollection);
      return true;
    }

    //! Add item to list.
    virtual bool AddToList(RCWrapAbstractC &list,RCWrapAbstractC &item) const {
      RCWrapC<CollectionC<EntryT> > lw(list,true);
      RavlAssert(lw.IsValid());
      EntryT value;
      if(!SystemTypeConverter().TypeConvert(item,value)) {
        RavlAssertMsg(0,"Item not compatible with collection. ");
        return false;
      }
      lw.Data().Append(value);
      return true;
    }

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneTypeListC<EntryT> > RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneTypeListC<EntryT> > ConstRefT;
  protected:
  };


  //! List containing sub gene's

  class GeneListC
   : public GeneC
  {
  public:
    //! Constuct from a geneType.
    GeneListC(const GeneTypeListBaseC &geneType,const std::vector<GeneC::ConstRefT> &aList);

    //! Factory constructor
    GeneListC(const XMLFactoryContextC &factory);

    //! Load form a binary stream
    GeneListC(BinIStreamC &strm);

    //! Load form a binary stream
    GeneListC(std::istream &strm);

    //! Save to binary stream
    virtual bool Save(BinOStreamC &strm) const;

    //! Save to binary stream
    virtual bool Save(std::ostream &strm) const;

    //! Save to binary stream
    virtual bool Save(RavlN::XMLOStreamC &strm) const;

    //! Access the list.
    const std::vector<GeneC::ConstRefT> &List() const
    { return m_list; }

    //! Generate an instance of the class.
    virtual void Generate(const GeneFactoryC &context,RCWrapAbstractC &handle) const;

    //! Visit all gene's in tree.
    virtual void Visit(GeneVisitorC &visitor) const;

    // Reference to this gene.
    typedef RavlN::SmartPtrC<GeneListC> RefT;

    // Const reference to this gene.
    typedef RavlN::SmartPtrC<const GeneListC> ConstRefT;
  protected:
    std::vector<GeneC::ConstRefT> m_list;
  };


   template<typename ClassT>
   class RegisterGeneListC
   {
   public:
     //static typename ClassT::RefT ConvertGeneFactory2Inst(const GeneFactoryC &factory)
     //{ return new ClassT(factory); }

     RegisterGeneListC(const char *nameOfType)
     {
       //RavlN::AddTypeName(typeid(ClassT),nameOfType);
       //m_refName = std::string(nameOfType) + "::RefT";
       //RavlN::AddTypeName(typeid(typename ClassT::RefT),m_refName.data());
       //RavlN::RegisterConversion(&ConvertGeneFactory2Inst);
     }

   protected:
     //std::string m_refName;
   };

}}
#endif