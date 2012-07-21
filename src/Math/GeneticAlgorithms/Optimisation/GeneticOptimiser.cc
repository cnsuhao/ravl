// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2011, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlGeneticOptimisation
//! author=Charles Galambos
//! docentry=Ravl.API.Math.Genetic.Optimisation

#include "Ravl/Genetic/GeneticOptimiser.hh"
#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/CallMethodPtrs.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/DP/FileFormatBinStream.hh"
#include "Ravl/IO.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define RAVL_CATCH_EXCEPTIONS 1

namespace RavlN { namespace GeneticN {


  GeneticOptimiserC::GeneticOptimiserC(const XMLFactoryContextC &factory)
   : m_mutationRate(static_cast<float>(factory.AttributeReal("mutationRate",0.2))),
     m_cross2mutationRatio(static_cast<float>(factory.AttributeReal("cross2mutationRatio",0.2))),
     m_keepFraction(static_cast<float>(factory.AttributeReal("keepFraction",0.3))),
     m_randomFraction(factory.AttributeReal("randomFraction",0.01)),
     m_populationSize(factory.AttributeUInt("populationSize",10)),
     m_numGenerations(factory.AttributeUInt("numGenerations",10)),
     m_terminateScore(static_cast<float>(factory.AttributeReal("teminateScore",-1.0))),
     m_createOnly(factory.AttributeBool("createOnly",false)),
     m_threads(factory.AttributeUInt("threads",1)),
     m_atWorkQueue(0),
     m_randomiseDomain(factory.AttributeBool("randomiseDomain",false)),
     m_runningAverageLength(factory.AttributeUInt("runningAverageLength",0))
  {
    factory.Attribute("logLevel",m_logLevel,RavlN::SYSLOG_INFO);
    RavlInfo("Setting debug level to '%s' ",RavlN::StringOf(m_logLevel).c_str());

    rThrowBadConfigContextOnFailS(factory,UseComponentGroup("StartPopulation",m_startPopulation,typeid(GenomeC)),"No start population");
    // Setting the fitness function via XML is optional
    factory.UseComponent("Fitness",m_evaluateFitness,true);
    if(!factory.UseComponent("GenePalette",m_genePalette,true,typeid(GenePaletteC))) {
      m_genePalette = new GenePaletteC();
    }
    RavlDebug("Mutation rate:%f Cross rate:%f Random:%f Keep:%f ",m_mutationRate,m_cross2mutationRatio,m_randomFraction,m_keepFraction);
  }

  //! Set fitness function to use
  void GeneticOptimiserC::SetFitnessFunction(EvaluateFitnessC &fitness) {
    m_evaluateFitness = &fitness;
  }

  //! Run whole optimisation
  void GeneticOptimiserC::Run() {
    if(!m_evaluateFitness.IsValid()) {
      RavlError("Not fitness function defined.");
      RavlAssertMsg(0,"No fitness function defined.");
      return ;
    }
    RavlDebug("Running %u generations ", m_numGenerations);
    MutexLockC lock(m_access);
    lock.Unlock();

    for(unsigned i = 0;i < m_numGenerations;i++) {
      float bestScore = 0;
      lock.Lock();
      if(!m_population.empty())
        bestScore = m_population.rbegin()->first;
      lock.Unlock();
      RavlInfo("Running generation %u  Initial score:%f ",i,bestScore);
      RunGeneration(i);
      lock.Lock();
      if(m_logLevel >= SYSLOG_DEBUG || 1) {
        m_population.rbegin()->second->Dump(RavlSysLog(SYSLOG_DEBUG));
      }
      if(m_terminateScore > 0 && m_population.rbegin()->first > m_terminateScore) {
        RavlDebug("Termination criteria met.");
        lock.Unlock();
        break;
      }
      lock.Unlock();
    }

    lock.Lock();
    if(m_population.empty()) {
      RavlInfo("Population list empty. ");
    } else {
      RavlInfo("Best final score %f ",m_population.rbegin()->first);
      OStreamC ostrm(std::cout);
      RavlN::XMLOStreamC outXML(ostrm);
      m_population.rbegin()->second->Save(outXML);
    }
    lock.Unlock();
  }

  //! Save population to file
  bool GeneticOptimiserC::SavePopulation(const StringC &filename) const
  {
    SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > population;
    ExtractPopulation(population);
    return RavlN::Save(filename,population,"abs",true);
  }

  //! Save population to file
  bool GeneticOptimiserC::LoadPopulation(const StringC &filename)
  {
    SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > population;
    if(!RavlN::Load(filename,population))
      return false;
    AddPopulation(population);
    return true;
  }

  //! Extract population from optimisers
  void GeneticOptimiserC::ExtractPopulation(SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > &population) const
  {
    RavlN::MutexLockC lock(m_access);
    if(m_randomiseDomain) {
      population = SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> >(m_currentSeeds.Size());
      for(unsigned i = 0;i < m_currentSeeds.Size();i++)
        population[i] = RavlN::Tuple2C<float,GenomeC::RefT>(0,m_currentSeeds[i]);
      return ;
    }
    size_t arraySize = m_population.size();
    population = SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> >(arraySize);
    unsigned i = 0;
    // Note: this saves worst to best, just in case you were thinking of truncating the list...
    for(std::multimap<float,GenomeC::RefT>::const_iterator it(m_population.begin());it != m_population.end();it++,i++)
    {
      population[i] = RavlN::Tuple2C<float,GenomeC::RefT>(it->first,it->second);
    }
    RavlAssert(i == arraySize);
  }


  //! Add population to optimisers, not this does not remove any entries already there
  void GeneticOptimiserC::AddPopulation(const SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > &population)
  {
    MutexLockC lock(m_access);
    for(unsigned i = 0;i < population.Size();i++) {
      m_population.insert(std::pair<const float,GenomeC::RefT>(population[i].Data1(),population[i].Data2()));
    }
    lock.Unlock();
  }

  //! Get the current best genome and its score.

  bool GeneticOptimiserC::GetBestGenome(GenomeC::RefT &genome,float &score) {
    MutexLockC lock(m_access);
    if(m_population.empty())
      return false;
    score = m_population.rbegin()->first;
    genome = m_population.rbegin()->second;
    return true;
  }


  //! Run generation.
  void GeneticOptimiserC::RunGeneration(UIntT generation)
  {
    RavlDebugIf(m_logLevel,"Examining results from last run. %s ",RavlN::StringOf(m_logLevel).c_str());
    unsigned count = 0;

    // Select genomes to be used as seeds for the next generation.
    unsigned numKeep = Floor(m_populationSize * m_keepFraction);
    if(numKeep >= m_populationSize)
      numKeep = m_populationSize - 1;
    if(numKeep < 1) numKeep = 1;
    // Compute the number of new genomes to create
    unsigned numCreate = m_populationSize - numKeep;

    CollectionC<GenomeC::RefT> seeds(numKeep);
    std::vector<GenomeC::RefT> newTestSet;
    //m_newTestSet.clear();
    newTestSet.reserve(m_populationSize + numKeep);

    MutexLockC lock(m_access);
    if(m_population.empty()) {
      RavlAssert(!m_startPopulation.empty());
      lock.Unlock();
      Evaluate(m_startPopulation);
      lock.Lock();
      if(m_population.empty()) {
        RavlError("Population empty.");
        return ;
      }
    }

    std::multimap<float,GenomeC::RefT>::reverse_iterator it(m_population.rbegin());
    while(it != m_population.rend() && count < numKeep) {
      seeds.Append(it->second);
      //RavlDebug(" Score:%f Age:%u Gen:%u Size:%zu @ %p ",it->first,m_population.rbegin()->second->Age(),it->second->Generation(),it->second->Size(),it->second.BodyPtr());
      if(m_randomiseDomain)
        newTestSet.push_back(it->second);
      it++;
      count++;
    }

    RavlDebugIf(m_logLevel,"Gen:%u Got %u seeds. Pop:%u Best score=%f Worst score=%f Best Age:%u Best Generation:%u ",
        generation,(UIntT) seeds.Size().V(),(UIntT) m_population.size(),(float) m_population.rbegin()->first,(float) m_population.begin()->first,(UIntT) m_population.rbegin()->second->Age(),(UIntT) m_population.rbegin()->second->Generation());

    if(m_randomiseDomain) {
      m_population.clear();
    } else {
      // Erase things we don't want to keep.
      if(it != m_population.rend()) {
        m_population.erase(m_population.begin(),it.base());
      }
    }
    m_currentSeeds = seeds.Array();
    lock.Unlock();

    unsigned noCrosses = Floor(numCreate * m_cross2mutationRatio);
    RavlDebugIf(m_logLevel,"Creating %d crosses. ",noCrosses);

    unsigned i = 0;
    // In the first generation there may not be enough seeds to make
    // sense doing this.
    if(seeds.Size() > 1) {
      for(;i < noCrosses;i++) {
        unsigned i1 = m_genePalette->RandomUInt32() % seeds.Size().V();
        unsigned i2 = m_genePalette->RandomUInt32() % seeds.Size().V();
        // Don't breed with itself.
        if(i1 == i2)
          i2 = (i1 + 1) % seeds.Size();
        GenomeC::RefT newGenome;
        seeds[i1]->Cross(*m_genePalette,*seeds[i2],newGenome);
        newGenome->SetGeneration(generation);
        newTestSet.push_back(newGenome);
      }
    }

    RavlDebugIf(m_logLevel,"Completing the population with mutation. %u (Random fraction %f) ", (UIntT) (m_populationSize - i),m_randomFraction);
    for(;i < m_populationSize;i++) {
      unsigned i1 = m_genePalette->RandomUInt32() % seeds.Size().V();
      GenomeC::RefT newGenome;
      if(Random1() < m_randomFraction) {
        ONDEBUG(RavlDebug("Random"));
        seeds[i1]->Mutate(*m_genePalette,1.0,newGenome);
      } else {
        ONDEBUG(RavlDebug("Mutate"));
        seeds[i1]->Mutate(*m_genePalette,m_mutationRate,newGenome);
      }
      newGenome->SetGeneration(generation);
      newTestSet.push_back(newGenome);
    }

    RavlDebugIf(m_logLevel,"Evaluating population size %s ",RavlN::StringOf(newTestSet.size()).data());
    // Evaluate the new genomes.
    Evaluate(newTestSet);
  }

  void GeneticOptimiserC::Evaluate(const std::vector<GenomeC::RefT> &pop)
  {
    MutexLockC lock(m_access);
    //std::swap(m_workQueue,pop);
    m_workQueue = pop;
    if(m_randomiseDomain) {
      m_evaluateFitness->GenerateNewProblem();
    }
    m_atWorkQueue = 0;
    lock.Unlock();
    if(m_threads == 1) {
      EvaluateWorker();
    } else {
      std::vector<LaunchThreadC> threads;
      threads.reserve(m_threads);
      for(unsigned i = 0;i < m_threads-1;i++) {
        threads.push_back(LaunchThread(TriggerPtr(RefT(*this),&GeneticOptimiserC::EvaluateWorker)));
      }
      // Use this thread too.
      EvaluateWorker();
      for(unsigned i = 0;i < threads.size();i++) {
        threads[i].WaitForExit();
      }
    }
  }

  void GeneticOptimiserC::EvaluateWorker() {
    EvaluateFitnessC::RefT evaluator = &dynamic_cast<EvaluateFitnessC &>(m_evaluateFitness->Copy());

    MutexLockC lock(m_access);
    GenePaletteC::RefT palette = m_genePalette.Copy();
    lock.Unlock();

    //RavlInfo("Palette has %u proxies. ",(unsigned) palette->ProxyMap().Size());
    while(1) {
      UIntT candidate;
      lock.Lock();
      candidate = m_atWorkQueue++;
      if(candidate >= m_workQueue.size())
        break;
      GenomeC::RefT genome = m_workQueue[candidate];
      lock.Unlock();
      float score = 0;
      if(!Evaluate(*evaluator,*genome,*palette,score))
        continue;
      if(m_runningAverageLength >= 1)
        score = genome->UpdateScore(score,m_runningAverageLength);
      lock.Lock();
      m_population.insert(std::pair<const float,GenomeC::RefT>(score,genome));
      lock.Unlock();
    }

  }

  //! Evaluate a single genome
  bool GeneticOptimiserC::Evaluate(EvaluateFitnessC &evaluator,
                                   const GenomeC &genome,
                                   GenePaletteC &palette,
                                   float &score)
  {
    GeneFactoryC factory(genome,palette);
    score = 0;
#if RAVL_CATCH_EXCEPTIONS
    try {
#endif
      RCWrapAbstractC anObj;
      factory.Get(anObj,evaluator.ObjectType());
      if(m_createOnly)
        return false;
      if(!evaluator.Evaluate(anObj,score))
        return false;
#if RAVL_CATCH_EXCEPTIONS
    } catch(std::exception &ex) {
      RavlWarning("Caught std exception '%s' evaluating agent.",ex.what());
      score = -1000000;
      //RavlAssert(0);
      return false;
    } catch(RavlN::ExceptionC &ex) {
      RavlWarning("Caught exception '%s' evaluating agent.",ex.what());
      score = -1000000;
      //RavlAssert(0);
      return false;
    } catch(...) {
      RavlWarning("Caught exception evaluating agent.");
      score = -1000000;
      RavlAssert(0);
      return false;
    }
#endif
    size_t size = genome.Size();
    //float sizeDiscount =  (size / 1000.0) * (0.5 + Random1()); //Floor(size / 10) * 0.01;
    float sizeDiscount = (size / 15) * 0.001f; //(AK) note integer division
    //float sizeDiscount = size / 1000.0;
    score -= sizeDiscount;
    return true;
  }


  void LinkGeneticOptimiser()
  {}

  static XMLFactoryRegisterC<GeneticOptimiserC> g_registerGeneticOptimiser("RavlN::GeneticN::GeneticOptimiserC");
  static RavlN::TypeNameC g_typeGeneticOptimiserRef(typeid(RavlN::GeneticN::GeneticOptimiserC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneticOptimiserC>");
  static RavlN::TypeNameC g_typeGeneticOptimiserState(typeid(SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > ),"RavlN::SArray1dC<RavlN::Tuple2C<float,RavlN::GeneticN::GenomeC::RefT>>");
  static FileFormatBinStreamC<SArray1dC<RavlN::Tuple2C<float,GenomeC::RefT> > > g_FileFormatBinStream_Array_Score_Genome("abs","Stuff");

}}
