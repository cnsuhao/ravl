#ifndef OMNI_OBJECT_ZMQGENETICOPTIMISER_HH
#define OMNI_OBJECT_ZMQGENETICOPTIMISER_HH 1

#include "Ravl/Genetic/GeneticOptimiser.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/Zmq/Socket.hh"

namespace RavlN {
  namespace ZmqN {

    /**
     * The genetic optimiser but uses Zmq to farm out evaluation of each candidate solution (i.e. Genome)
     */

    class ZmqGeneticOptimiserC : public RavlN::GeneticN::GeneticOptimiserC
    {
    public:
      /**
       * Factory constructor
       * @param factory The factory
       */
      ZmqGeneticOptimiserC(const RavlN::XMLFactoryContextC & factory);

      //! Handle to optimiser
      typedef RavlN::SmartPtrC<ZmqGeneticOptimiserC> RefT;

    protected:

      /**
       * The function that takes the entire population and evaluates them
       * @param pop The Genome population
       */
      virtual bool Evaluate(const std::vector<RavlN::GeneticN::GenomeC::RefT> &pop);


      SocketC::RefT m_sender; //!< Send out tasks
      SocketC::RefT m_receiver; //!< Receive results

    };

  }
}

#endif
