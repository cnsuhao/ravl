// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id: DesignCascade.cc 7590 2010-02-23 12:03:11Z kier $"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Classify/DesignCascade.cc"

#include "Ravl/PatternRec/DesignCascade.hh"
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
  
  DesignCascadeBodyC::DesignCascadeBodyC(const DesignClassifierSupervisedC & design) :
      m_design(design)
  {
  }
  
  DesignCascadeBodyC::DesignCascadeBodyC(const XMLFactoryContextC & factory) :
      DesignClassifierSupervisedWithValidationBodyC(factory)
  {
    if (!factory.UseComponent("Design", m_design))
      RavlIssueError("No Design Classifier in XML factory");
  }
  
  //: Load from stream.
  
  DesignCascadeBodyC::DesignCascadeBodyC(std::istream &strm) :
      DesignClassifierSupervisedWithValidationBodyC(strm)
  {
    int version;
    strm >> version;
    if (version != 0)
      throw ExceptionOutOfRangeC("DesignCascadeBodyC::DesignCascadeBodyC(std::istream &), Unrecognised version number in stream. ");
    strm >> m_design;
  }
  
  //: Load from binary stream.
  
  DesignCascadeBodyC::DesignCascadeBodyC(BinIStreamC &strm) :
      DesignClassifierSupervisedWithValidationBodyC(strm)
  {
    int version;
    strm >> version;
    if (version != 0)
      throw ExceptionOutOfRangeC("DesignCascadeBodyC::DesignCascadeBodyC(BinIStreamC &), Unrecognised version number in stream. ");
    strm >> m_design;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignCascadeBodyC::Save(std::ostream &out) const
  {
    if (!DesignClassifierSupervisedBodyC::Save(out))
      return false;
    int version = 0;
    out << ' ' << version << ' ' << m_design;
    return true;
  }
  
  //: Writes object to stream, can be loaded using constructor
  
  bool DesignCascadeBodyC::Save(BinOStreamC &out) const
  {
    if (!DesignClassifierSupervisedBodyC::Save(out))
      return false;
    int version = 0;
    out << version << m_design;
    return true;
  }
  
  ClassifierC DesignCascadeBodyC::Apply(const DataSetVectorLabelC & trainingSet,
      const DataSetVectorLabelC & validationSet,
      const FeatureSelectorC & featureSelection,
      const ErrorC & error)
  {
    RavlInfo("Designing Cascade Classifier!");
    RavlAssertMsg(trainingSet.Size() == trainingSet.Size(),
        "DesignCascadeBodyC::Apply(), Sample of vector and labels should be the same size.");

    SampleVectorC designVectors = trainingSet.Sample1();
    SampleLabelC designLabels = trainingSet.Sample2();

    SampleVectorC validationVectors = validationSet.Sample1();
    SampleLabelC validationLabels = validationSet.Sample2();

    UIntT maxStages = 100;
    RealT targetFA = 0.001; // 0.1%
    SampleLabelC sampleLabel(trainingSet.Sample2());
    SArray1dC<UIntT> labelSums = sampleLabel.LabelSums();
    UIntT targetCount = Round((RealT) labelSums[1] * targetFA);

    CollectionC<ClassifierC> classifiers(maxStages);
    CollectionC<RealT> thresholds(maxStages);
    CollectionC<FuncSubsetC> features(maxStages);

    for (UIntT i = 0; i < maxStages; i++) {

      SizeT numberOfDimensions = designVectors.First().Size();

      //FeatureSelectPlusLMinusRC featureSelection(2, 1, 5);
      //FeatureSelectAsymmetricAdaBoostC featureSelection(0.5, 10);
      ClassifierC bestClassifier;
      SArray1dC<IndexC> selectedFeatures = featureSelection.SelectFeatures(m_design,
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
      RealT falseRejectRate = 0.001; // we want a low false rejection rate - we are prepared to have lots of false accepts
      RealT falseAcceptRate = errorBinaryClassifier.FalseAcceptRate(bestClassifier, dset, falseRejectRate, threshold);

      RavlInfo("Stage %d classifier, FA: %0.4f FR %0.4f at Threshold %0.4f", i, falseAcceptRate, falseRejectRate, threshold);
      thresholds.Append(threshold);

      /*
       * Now we need to find data that is not correctly classified
       */
      SampleVectorC remIn(validationVectors.Size());
      SampleLabelC remOut(validationLabels.Size());
      SArray1dC<UIntT>labelSums(2);
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
      SArray1dC<RealT>labelErrors = error.ErrorByLabel(cascade, validationSet);
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
  RavlN::XMLFactoryRegisterHandleConvertC<DesignCascadeC, DesignClassifierSupervisedWithValidationC> g_registerXMLFactoryDesignCascade("RavlN::DesignCascadeC");

  RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignCascadeBodyC, DesignCascadeC, DesignClassifierSupervisedWithValidationC);

  void linkDesignCascade()
  {
  }

}
