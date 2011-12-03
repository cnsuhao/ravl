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
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/CallMethodPtrs.hh"
#include "Ravl/OS/SysLog.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN { namespace GeneticN {


  GeneticOptimiserC::GeneticOptimiserC(const XMLFactoryContextC &factory)
   : m_mutationRate(static_cast<float>(factory.AttributeReal("mutationRate",0.1))),
     m_crossRate(static_cast<float>(factory.AttributeReal("crossRate",0.2))),
     m_keepFraction(static_cast<float>(factory.AttributeReal("keepFraction",0.3))),
     m_randomFraction(static_cast<float>(factory.AttributeReal("randomFraction",0.01))),
     m_populationSize(factory.AttributeUInt("populationSize",10)),
     m_numGenerations(factory.AttributeUInt("numGenerations",10)),
     m_runLengthVirtual(static_cast<float>(factory.AttributeReal("runLengthVirtual",10.0))),
     m_runLengthCPU(static_cast<float>(factory.AttributeReal("runLengthCPU",10.0))),
     m_terminateScore(static_cast<float>(factory.AttributeReal("teminateScore",-1.0))),
     m_createOnly(factory.AttributeBool("createOnly",false)),
     m_threads(factory.AttributeUInt("threads",1))
  {
    rThrowBadConfigContextOnFailS(factory,UseComponentGroup("StartPopulation",m_startPopulation,typeid(GenomeC)),"No start population");
    // Setting the fitness function via XML is optional
    factory.UseComponent("Fitness",m_evaluateFitness,true);
  }

  //! Set fitness function to use
  void GeneticOptimiserC::SetFitnessFunction(EvaluateFitnessC &fitness) {
    m_evaluateFitness = &fitness;
  }

  //! Run whole optimisation
  void GeneticOptimiserC::Run() {
    if(!m_evaluateFitness.IsValid()) {
      RavlSysLogf(SYSLOG_ERR,"Not fitness function defined.");
      RavlAssertMsg(0,"No fitness function defined.");
      return ;
    }
    for(unsigned i = 0;i < m_numGenerations;i++) {
      RavlSysLogf(SYSLOG_INFO,"Running generation %u ",i);
      RunGeneration(i);
      if(m_terminateScore > 0 && m_population.rbegin()->first > m_terminateScore)
        break;
    }
    RavlSysLogf(SYSLOG_INFO,"Best final score %f ",m_population.rbegin()->first);
    OStreamC ostrm(std::cout);
    RavlN::XMLOStreamC outXML(ostrm);
    m_population.rbegin()->second->Save(outXML);
  }

  //! Run generation.
  void GeneticOptimiserC::RunGeneration(UIntT generation)
  {
    if(m_population.empty()) {
      RavlSysLogf(SYSLOG_DEBUG,"Ranking initial population");
      RavlAssert(!m_startPopulation.empty());
      Evaluate(m_startPopulation);
      return ;
    }

    RavlSysLogf(SYSLOG_DEBUG,"Examining results from last run. ");
    unsigned count = 0;
    std::multimap<float,GenomeC::RefT>::reverse_iterator it(m_population.rbegin());

    // Select genomes to be used as seeds for the next generation.
    std::vector<GenomeC::RefT> seeds;
    unsigned numKeep = Floor(m_populationSize * m_keepFraction);
    if(numKeep < 1) numKeep = 1;
    seeds.reserve(numKeep);

    if(m_population.empty()) {
      RavlSysLogf(SYSLOG_ERR,"Population empty.");
      return ;
    }

    while(it != m_population.rend() && count < numKeep) {
      seeds.push_back(it->second);
      //RavlSysLogf(SYSLOG_DEBUG," Score:%f Age:%u Gen:%u Size:%zu @ %p ",it->first,m_population.rbegin()->second->Age(),it->second->Generation(),it->second->Size(),it->second.BodyPtr());
      it++;
      count++;
    }

    // Erase things we don't want to keep.
    if(it != m_population.rend()) {
      m_population.erase(m_population.begin(),it.base());
    }

    RavlSysLogf(SYSLOG_DEBUG,"Got %u seeds. Pop:%u Best score=%f Age:%u Generation:%u ",(UIntT) seeds.size(),(UIntT) m_population.size(),(float) m_population.rbegin()->first,(UIntT) m_population.rbegin()->second->Age(),(UIntT) m_population.rbegin()->second->Generation());

    std::vector<GenomeC::RefT> newTestSet;
    newTestSet.reserve(m_populationSize);

    unsigned noCrosses = Floor(m_populationSize * m_crossRate);
    RavlSysLogf(SYSLOG_DEBUG,"Creating %d crosses. ",noCrosses);

    unsigned i = 0;
    // In the first generation there may not be enough seeds to make
    // sense doing this.
    if(seeds.size() > 1) {
      for(;i < noCrosses;i++) {
        unsigned i1 = RandomInt() % seeds.size();
        unsigned i2 = RandomInt() % seeds.size();
        // Don't breed with itself.
        if(i1 == i2)
          i2 = (i1 + 1) % seeds.size();
        GenomeC::RefT newGenome;
        seeds[i1]->Cross(*seeds[i2],newGenome);
        newGenome->SetGeneration(generation);
        newTestSet.push_back(newGenome);
      }
    }

    RavlSysLogf(SYSLOG_DEBUG,"Completing the population with mutation. %u (Random fraction %f) ", (UIntT) (m_populationSize - i),m_randomFraction);
    for(;i < m_populationSize;i++) {
      unsigned i1 = RandomInt() % seeds.size();
      GenomeC::RefT newGenome;
      if(Random1() < m_randomFraction) {
        ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Random"));
        seeds[i1]->Mutate(1.0,newGenome);
      } else {
        ONDEBUG(RavlSysLogf(SYSLOG_DEBUG,"Mutate"));
        seeds[i1]->Mutate(m_mutationRate,newGenome);
      }
      newGenome->SetGeneration(generation);
      newTestSet.push_back(newGenome);
    }

    RavlSysLogf(SYSLOG_DEBUG,"Evaluating population");
    // Evaluate the new genomes.
    Evaluate(newTestSet);
  }

  void GeneticOptimiserC::Evaluate(const std::vector<GenomeC::RefT> &pop)
  {
    MutexLockC lock(m_access);
    //std::swap(m_workQueue,pop);
    m_workQueue = pop;

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
    while(1) {
      UIntT candidate;
      MutexLockC lock(m_access);
      candidate = m_atWorkQueue++;
      if(candidate >= m_workQueue.size())
        break;
      GenomeC::RefT genome = m_workQueue[candidate];
      lock.Unlock();
      float score = 0;
      if(!Evaluate(*evaluator,*genome,score))
        continue;
      lock.Lock();
      m_population.insert(std::pair<const float,GenomeC::RefT>(score,genome));
      lock.Unlock();
    }

  }

  //! Evaluate a single genome
  bool GeneticOptimiserC::Evaluate(EvaluateFitnessC &evaluator,const GenomeC &genome,float &score) {
    GeneFactoryC factory(genome);
    score = 0;
    try {
      RCWrapAbstractC anObj;
      factory.Get(anObj,evaluator.ObjectType());
      if(m_createOnly)
        return false;
      if(!evaluator.Evaluate(anObj,score))
        return false;
    } catch(...) {
      RavlSysLogf(SYSLOG_WARNING,"Failed to evaluate agent.");
      RavlAssert(0);
      return false;
    }
    size_t size = genome.Size();
    //float sizeDiscount =  (size / 1000.0) * (0.5 + Random1()); //Floor(size / 10) * 0.01;
    float sizeDiscount = (size / 15) * 0.001f; //(AK) note integer division
    //float sizeDiscount = size / 1000.0;
    score -= sizeDiscount;
    return true;
  }


  void LinkGeneticOptimiser()
  {}

  XMLFactoryRegisterC<GeneticOptimiserC> g_registerGeneticOptimiser("RavlN::GeneticN::GeneticOptimiserC");
  static RavlN::TypeNameC g_typeEnvironmentSimpleState(typeid(RavlN::GeneticN::GeneticOptimiserC::RefT),"RavlN::SmartPtrC<RavlN::GeneticN::GeneticOptimiserC>");

}}
