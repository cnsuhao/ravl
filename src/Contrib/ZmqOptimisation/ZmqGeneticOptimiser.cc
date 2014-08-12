#include "Ravl/Zmq/ZmqGeneticOptimiser.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/RLog.hh"
#include "Ravl/Threads/LaunchThread.hh"
#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/Point2d.hh"
#include "Ravl/DP/Blackboard.hh"
#include "Ravl/DP/FileFormatStream.hh"
#include "Ravl/OS/Date.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

#define RAVL_CATCH_EXCEPTIONS 1

namespace RavlN {
  namespace ZmqN {

    using namespace RavlN;
    using namespace RavlN::GeneticN;

    ZmqGeneticOptimiserC::ZmqGeneticOptimiserC(const XMLFactoryContextC &factory) :
        GeneticOptimiserC(factory)
    {
      rThrowBadConfigContextOnFailS(factory, UseComponent("Sender", m_sender), "Failed to find Sender socket in XML!");
      rThrowBadConfigContextOnFailS(factory, UseComponent("SinkReceiver", m_receiver), "Failed to find SinkReceiver socket in XML!");
    }

    bool ZmqGeneticOptimiserC::Evaluate(const std::vector<GenomeC::RefT> &pop)
    {
      MutexLockC lock(m_access);
      if (m_randomiseDomain) {
        // FIXME: Speak to CG.  I could be in trouble here as I do not have a valid fitness function!
        RavlAssertMsg(0, "Do not have a fitness function!");
        m_evaluateFitness->GenerateNewProblem();
      }
      lock.Unlock();

      rInfo("Evaluating a new generation with %s genomes.", StringOf(pop.size()).data());

      for (IntT i = 0; i < (IntT) pop.size(); i++) {

        // Need to get next point for evaluation from the GeneFactory
        GenomeC::RefT genome = pop[i];
        GenePaletteC::RefT palette = m_genePalette.Copy();
        GeneFactoryC factory(*genome, *palette);

        // For this we always assume the thing to be 'optimised' is a Blackboard
        RCWrapAbstractC anObj;
        factory.Get(anObj, typeid(BlackboardC));
        BlackboardC *bb;
        anObj.GetPtr(bb);

        // We want to add an id for the worker
        bb->Put("id", i);

        BufOStreamC bos;
        {
          BinOStreamC os(bos);
          os << *bb;
        }

        // OK I am going to send the data into the ether
        if (!m_sender->Send(bos.Data())) {
          rWarning("Trouble sending data for evaluation!");
          continue;
        }
        RavlN::Sleep(0.1);

        // and move on to the next data point to be evaluated
      }

      bool allDone = false;

      // Now we have to wait for all the workers.
      // At the moment I am just counting the number of results.
      UIntT done = 0;
      FloatT bestScore = 0.0;

      while (!allDone) {

        SArray1dC<char> data;
        if (!m_receiver->Recieve(data)) {
          rWarning("Trouble receiving data");
          continue;
        }
        BufIStreamC bis(data);
        BinIStreamC is(bis);
        BlackboardC bb;
        is >> bb;

        IntT id = 0;
        RealT score = 0.0f;
        if (!bb.Get("id", id)) {
          rError("Need an id");
          continue;
        }

        if (!bb.Get("score", score)) {
          rError("Need a score");
          continue;
        }

        GenomeC::RefT genome = pop[id];

        size_t size = genome->Size();
        //float sizeDiscount =  (size / 1000.0) * (0.5 + Random1()); // Floor(size / 10) * 0.01;
        float sizeDiscount = ((float) size / 15.0f) * 0.001f;
        //float sizeDiscount = size / 1000.0;
        score -= sizeDiscount;

        lock.Lock();
        if (m_runningAverageLength >= 1)
          score = genome->UpdateScore(score, m_runningAverageLength);
        m_population.insert(std::pair<const float, GenomeC::RefT>(score, genome));
        lock.Unlock();
        done++;

        StringC name;
        if (bb.Get("name", name)) {
          rInfo("Worker Id: %d Name: %s Score: %0.6f", id, name.data(), score);
        } else {
          rInfo("Worker Id: %d has score %0.6f", id, score);
        }

        if(score > bestScore) {
          bestScore = score;
          rWarning("New best score for generation %0.6f", bestScore);
        }

        // Have we got all the results for the current generation?
        if (done == pop.size()) {
          allDone = true;
          rInfo("Current generation finished!");
        }

      }

      return true;
    }

    void LinkGeneticOptimiser()
    {
    }

    static XMLFactoryRegisterConvertC<ZmqGeneticOptimiserC, GeneticOptimiserC> g_registerConvertZmqGeneticOptimiser("RavlN::ZmqN::ZmqGeneticOptimiserC");
    static RavlN::TypeNameC g_typeZmqGeneticOptimiserRef(typeid(RavlN::ZmqN::ZmqGeneticOptimiserC::RefT),
        "RavlN::SmartPtrC<RavlN::GeneticN::GeneticOptimiserC>");

  }
}
