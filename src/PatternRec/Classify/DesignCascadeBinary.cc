// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id: DesignCascadeBinary.cc 7590 2010-02-23 12:03:11Z kier $"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Classify/DesignCascadeBinary.cc"

#include "Ravl/PatternRec/DesignCascadeBinary.hh"
#include "Ravl/PatternRec/ClassifierOneAgainstAll.hh"
#include "Ravl/VirtualConstructor.hh"
#include "Ravl/BinStream.hh"
#include "Ravl/PatternRec/DataSetVectorLabel.hh"
#include "Ravl/SArray1dIter2.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/Collection.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/PatternRec/DataSet3Iter.hh"
#include "Ravl/PatternRec/FuncSubset.hh"
#include "Ravl/Face/Roc.hh"
#include "Ravl/PatternRec/ClassifierCascade.hh"
#include "Ravl/PatternRec/Error.hh"
#include "Ravl/PatternRec/FeatureSelectPlusLMinusR.hh"
#include "Ravl/PatternRec/FeatureSelectAsymmetricAdaBoost.hh"
#include "Ravl/PatternRec/ErrorBinaryClassifier.hh"

namespace RavlN {
  
  //: Constructor.
  
  DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(const DesignClassifierSupervisedC & design,
      const FeatureSelectorC & featureSelector,
      UIntT maxStages,
      RealT targetFalseRejection) :
      DesignCascadeBodyC(design), m_featureSelector(featureSelector), m_maxStages(maxStages), m_targetFalseRejection(targetFalseRejection)
  {
  }
  
  DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(const XMLFactoryContextC & factory) :
      DesignCascadeBodyC(factory),
      m_maxStages(factory.AttributeInt("maxStages", 10)),
      m_targetFalseRejection(factory.AttributeReal("targetFalseRejection", 0.001))
  {
    if(!factory.UseComponent("FeatureSelection", m_featureSelector)) {
      throw ExceptionC("Unable to load feature selector from XML config file.");
    }
  }
  
  //: Load from stream.
  
  DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(std::istream &strm) :
      DesignCascadeBodyC(strm)
  {
    int version;
    strm >> version;
    if (version != 0)
      throw ExceptionOutOfRangeC("DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(std::istream &), Unrecognised version number in stream. ");
    strm >> m_featureSelector >> m_maxStages >> m_targetFalseRejection;
  }
  
  //: Load from binary stream.
  
  DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(BinIStreamC &strm) :
      DesignCascadeBodyC(strm)
  {
    int version;
    strm >> version;
    if (version != 0)
      throw ExceptionOutOfRangeC("DesignCascadeBinaryBodyC::DesignCascadeBinaryBodyC(BinIStreamC &), Unrecognised version number in stream. ");
    strm >> m_featureSelector >> m_maxStages >> m_targetFalseRejection;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignCascadeBinaryBodyC::Save(std::ostream &out) const
  {
    if (!DesignCascadeBodyC::Save(out))
      return false;
    int version = 0;
    out << ' ' << version << ' ' << m_featureSelector << ' ' << m_maxStages << ' ' << m_targetFalseRejection;
    return true;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignCascadeBinaryBodyC::Save(BinOStreamC &out) const
  {
    if (!DesignCascadeBodyC::Save(out))
      return false;
    int version = 0;
    out << version << m_featureSelector << m_maxStages << m_targetFalseRejection;
    return true;
  }
  
  ClassifierC DesignCascadeBinaryBodyC::Apply(const DataSetVectorLabelC & trainingSet,
      const DataSetVectorLabelC & validationSet)
  {
    RavlInfo("Designing Cascade Classifier!");
    RavlAssertMsg(trainingSet.Size() == trainingSet.Size(),
        "DesignCascadeBinaryBodyC::Apply(), Sample of vector and labels should be the same size.");
    RavlAssertMsg(trainingSet.Sample2().LabelSums().Size() == 2, "DesignCascadeBinaryBodyC::Apply() only supports two classes.");

    RavlInfo("Starting to design CascadeBinary classifier with stages %d and target FR %0.4f", m_maxStages, m_targetFalseRejection);

    SampleVectorC designVectors = trainingSet.Sample1();
    SampleLabelC designLabels = trainingSet.Sample2();

    SampleVectorC validationVectors = validationSet.Sample1();
    SampleLabelC validationLabels = validationSet.Sample2();

    SArray1dC<UIntT> labelSums = validationLabels.LabelSums();
    UIntT targetCount = Round((RealT) labelSums[1] * m_targetFalseRejection);

    CollectionC<ClassifierC> classifiers(m_maxStages);
    CollectionC<RealT> thresholds(m_maxStages);
    CollectionC<FuncSubsetC> features(m_maxStages);

    for (UIntT i = 0; i < m_maxStages; i++) {

      SizeT numberOfDimensions = designVectors.First().Size();

      //FeatureSelectPlusLMinusRC featureSelection(2, 1, 5);
      //FeatureSelectAsymmetricAdaBoostC featureSelection(0.5, 10);
      ClassifierC bestClassifier;
      SArray1dC<IndexC> selectedFeatures = m_featureSelector.SelectFeatures(m_design,
          DataSetVectorLabelC(designVectors, designLabels),
          DataSetVectorLabelC(validationVectors, validationLabels),
          bestClassifier);
      FuncSubsetC funcSubset(selectedFeatures, numberOfDimensions);
      ///bestClassifier = ClassifierPreprocessC(FuncSubsetC(selectedFeatures, numberOfDimensions), bestClassifier);
      classifiers.Append(bestClassifier);
      features.Append(funcSubset);

      /*
       * We have designed our classifier and we want to find the threshold which gives us no
       * false rejections
       */
      ErrorBinaryClassifierC errorBinaryClassifier;

      DataSetVectorLabelC dset(SampleVectorC(validationVectors, selectedFeatures), validationLabels);
      RealT threshold;
      RealT falseAcceptRate = errorBinaryClassifier.FalseAcceptRate(bestClassifier, dset, m_targetFalseRejection, threshold);

      RavlInfo("Stage %d classifier, FA: %0.4f FR %0.4f at Threshold %0.4f", i, falseAcceptRate, m_targetFalseRejection, threshold);
      thresholds.Append(threshold);

      /*
       * Now we need to find data that is not correctly classified
       */
      SampleVectorC remIn(validationVectors.Size());
      SampleLabelC remOut(validationLabels.Size());
      SArray1dC<UIntT> labelSums(2);
      labelSums.Fill(0);
      for (DataSet2IterC<SampleVectorC, SampleLabelC> it(validationVectors, validationLabels); it; it++) {

        RealT score = bestClassifier.LabelProbability(funcSubset.Apply(it.Data1()), 0);

        //RavlInfo("%0.4f %d", score, it.Data2());
        if (it.Data2() == 0 && score >= threshold) {
          remIn.Append(it.Data1());
          remOut.Append(it.Data2());
          labelSums[it.Data2()]++;
        }

        else if (it.Data2() == 1 && score > threshold) {
          remIn.Append(it.Data1());
          remOut.Append(it.Data2());
          labelSums[it.Data2()]++;
        }

      }

      RavlInfo("Label 0 '%d' Label 1 '%d', Target Label 1 '%d' ", labelSums[0], labelSums[1], targetCount);

      /*
       * How is the Cascade getting on?
       */
      ClassifierCascadeC cascade(classifiers.SArray1d(), thresholds.SArray1d(), features.SArray1d());
      ErrorC error;
      SArray1dC<RealT> labelErrors = error.ErrorByLabel(cascade, validationSet);
      RavlInfo("Cascade errors False Rejection %0.4f False Acceptance %0.4f", labelErrors[0], labelErrors[1]);

      if (labelSums[1] <= targetCount) {
        RavlInfo("Training complete!");
        break;
      }
      
      validationVectors = remIn;
      validationLabels = remOut;

    }

    return ClassifierCascadeC(classifiers.SArray1d(), thresholds.SArray1d(), features.SArray1d());

  }
  //: Create a classifier from training and validation set
  
  //////////////////////////////////////////////////////////
  RavlN::XMLFactoryRegisterHandleConvertC<DesignCascadeBinaryC, DesignCascadeC> g_registerXMLFactoryDesignCascadeBinary("RavlN::DesignCascadeBinaryC");

  RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignCascadeBinaryBodyC, DesignCascadeBinaryC, DesignCascadeC);

  void linkDesignCascadeBinary()
  {
  }

}
