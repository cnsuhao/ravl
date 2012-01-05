
#include "Ravl/Genetic/GPVariable.hh"
#include "Ravl/Genetic/GPInstruction.hh"
#include "Ravl/UnitTest.hh"

int testGeneInstIO();

int main(int nargs,char **argv)
{
  int ln = 0;
  if((ln = testGeneInstIO()) != 0) {
    std::cerr <<"Test failed on line " << ln << "\n";
    return 1;
  }
  return 0;
}

using RavlN::GeneticN::GenomeC;
using RavlN::GeneticN::GeneFactoryC;
using RavlN::GeneticN::GPInstructionC;

int testGeneInstIO()
{
  for(unsigned i = 0;i < 10000;i++) {

    GenomeC::RefT genome = new GenomeC(*RavlN::GeneticN::InstructionGeneType());

    // Instantiate genome
    GeneFactoryC factory(*genome);
    GPInstructionC::RefT inst;
    factory.Get(inst);
    RAVL_TEST_TRUE(inst.IsValid());

    GenomeC::RefT genomeRL;
    if(!TestBinStreamIO(genome,genomeRL))
      return __LINE__;
  }
  return 0;
}
