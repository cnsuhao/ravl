

#include "Ravl/Genetic/Genome.hh"
#include "Ravl/Genetic/GenomeConst.hh"
#include "Ravl/Genetic/GenomeClass.hh"

#include "Ravl/UnitTest.hh"
#include "Ravl/OS/SysLog.hh"

int testGenomeIO();
int testGeneIntIO();
int testGeneFloatIO();
int testGeneClassIO();

int main(int nargs,char **argv)
{
  RavlN::SysLogOpen("testGeneticOpt");
  int ln;

  if((ln = testGeneIntIO()) != 0) {
    RavlError("Unit test failed on line %d ",ln);
    return 1;
  }

  if((ln = testGeneFloatIO()) != 0) {
    RavlError("Unit test failed on line %d ",ln);
    return 1;
  }

  if((ln = testGeneClassIO()) != 0) {
    RavlError("Unit test failed on line %d ",ln);
    return 1;
  }

  if((ln = testGenomeIO()) != 0) {
    RavlError("Unit test failed on line %d ",ln);
    return 1;
  }

  RavlInfo("Test completed ok. ");
  return 0;
}

using RavlN::GeneticN::GenomeC;
using RavlN::GeneticN::GeneTypeIntC;
using RavlN::GeneticN::GeneIntC;
using RavlN::GeneticN::GeneTypeFloatC;
using RavlN::GeneticN::GeneFloatC;
using RavlN::GeneticN::GeneTypeClassC;
using RavlN::GeneticN::GeneClassC;

int testGeneIntIO()
{
  GeneTypeIntC::RefT geneType = new GeneTypeIntC("igloo",1,10);
  GeneTypeIntC::RefT geneTypeRL;

  if(!TestBinStreamIO(geneType,geneTypeRL))
    return __LINE__;

  RAVL_TEST_EQUALS(geneType->Name(),geneTypeRL->Name());
  RAVL_TEST_EQUALS(geneType->Min(),geneTypeRL->Min());
  RAVL_TEST_EQUALS(geneType->Max(),geneTypeRL->Max());

  GeneIntC::RefT gene = new GeneIntC(*geneType,5);
  GeneIntC::RefT geneRL;

  if(!TestBinStreamIO(gene,geneRL))
    return __LINE__;

  RAVL_TEST_EQUALS(gene->Value(),geneRL->Value());
  RAVL_TEST_EQUALS(gene->Type().Name(),geneType->Name());

  return 0;
}

int testGeneFloatIO()
{
  GeneTypeFloatC::RefT geneType = new GeneTypeFloatC("bannana",1.0,10.0);
  GeneTypeFloatC::RefT geneTypeRL;

  if(!TestBinStreamIO(geneType,geneTypeRL))
    return __LINE__;

  RAVL_TEST_EQUALS(geneType->Name(),geneTypeRL->Name());
  RAVL_TEST_EQUALS(geneType->Min(),geneTypeRL->Min());
  RAVL_TEST_EQUALS(geneType->Max(),geneTypeRL->Max());

  GeneFloatC::RefT gene = new GeneFloatC(*geneType,5.0);
  GeneFloatC::RefT geneRL;

  if(!TestBinStreamIO(gene,geneRL))
    return __LINE__;

  RAVL_TEST_EQUALS(gene->Value(),geneRL->Value());
  RAVL_TEST_EQUALS(gene->Type().Name(),geneType->Name());

  return 0;
}

int testGeneClassIO()
{
  GeneTypeClassC::RefT geneType = new GeneTypeClassC(typeid(RavlN::GeneticN::GeneFactoryC));
  GeneTypeClassC::RefT geneTypeRL;

  if(!TestBinStreamIO(geneType,geneTypeRL))
    return __LINE__;

  if(geneType->TypeInfo() != geneTypeRL->TypeInfo())
    return __LINE__;
  RAVL_TEST_EQUALS(geneType->TypeName(),geneTypeRL->TypeName());

  GeneClassC::RefT gene = new GeneClassC(*geneType);
  GeneClassC::RefT geneRL;

  if(!TestBinStreamIO(gene,geneRL))
    return __LINE__;

  RAVL_TEST_EQUALS(gene->Type().Name(),geneType->Name());

  return 0;
}



int testGenomeIO()
{
  GeneTypeIntC::RefT geneType = new GeneTypeIntC("igloo",1,10);
  GeneIntC::RefT gene = new GeneIntC(*geneType,5);
  GenomeC::RefT genome = new GenomeC(*gene);

  GenomeC::RefT genomeReconstructed;

  if(!TestBinStreamIO(genome,genomeReconstructed))
    return __LINE__;

  RAVL_TEST_EQUALS(genome->Age(),genomeReconstructed->Age());
  RAVL_TEST_EQUALS(genome->RootGene().Type().Name(),geneType->Name());

  return 0;
}
