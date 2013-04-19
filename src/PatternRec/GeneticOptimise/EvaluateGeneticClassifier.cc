
#include "Ravl/PatternRec/EvaluateGeneticClassifier.hh"

#define DODEBUG	0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlN {

  //! Constructor
  EvaluateGeneticClassiferC::EvaluateGeneticClassiferC()
  {}

  //! Constructor
  EvaluateGeneticClassiferC::EvaluateGeneticClassiferC(const ErrorC &errorMeasure,
                            const DataSetVectorLabelC &trainSample,
                            const DataSetVectorLabelC &crossValidateSample)
   : m_errorMeasure(errorMeasure),
     m_trainSample(trainSample),
     m_crossValidateSample(crossValidateSample)
  {}

  //! Copy constructor
  EvaluateGeneticClassiferC::EvaluateGeneticClassiferC(const EvaluateGeneticClassiferC &other)
    : m_errorMeasure(other.m_errorMeasure.Copy()),
      m_trainSample(other.m_trainSample),
      m_crossValidateSample(other.m_crossValidateSample)
  {}

  //! Copy me.
  RavlN::RCBodyVC &EvaluateGeneticClassiferC::Copy() const
  { return *new EvaluateGeneticClassiferC(*this); }

  //! Access type of object evaluated for fitness.
  const std::type_info &EvaluateGeneticClassiferC::ObjectType() const
  { return typeid(DesignClassifierSupervisedC); }

  //! Evaluate the fit
  bool EvaluateGeneticClassiferC::Evaluate(RCWrapAbstractC &obj,float &score)
  {
    RavlN::RCWrapC<DesignClassifierSupervisedC> wrap(obj,true);
    if(!wrap.IsValid())
      return false;
    DesignClassifierSupervisedC designer = wrap.Data();
    ClassifierC classifier = designer.Apply(m_trainSample.Sample1(),m_trainSample.Sample2());
    RealT cost = m_errorMeasure.Error(classifier,m_crossValidateSample);

    score = 1.0/ (cost + 0.0000001);
    ONDEBUG(RavlDebug(" Cost:%f Score:%f ",cost,score));
    return true;
  }

}
