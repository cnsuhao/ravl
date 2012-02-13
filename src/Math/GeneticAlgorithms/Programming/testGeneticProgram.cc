
#include "Ravl/Genetic/GPVariable.hh"
#include "Ravl/Genetic/GPInstruction.hh"
#include "Ravl/Genetic/GeneFactory.hh"
#include "Ravl/UnitTest.hh"
#include "Ravl/OS/SysLog.hh"

int testGeneInstIO();

int main(int nargs,char **argv)
{
//  RavlN::SysLogOpen("testGeneticOpt",false,true,false,-1,true);

  RavlInfo("Starting test. ");
  RAVL_RUN_TEST(testGeneInstIO());

  RavlInfo("Test passed ok. ");
  return 0;
}

using RavlN::GeneticN::GeneC;
using RavlN::GeneticN::GenePaletteC;
using RavlN::GeneticN::GenomeC;
using RavlN::GeneticN::GeneFactoryC;
using RavlN::GeneticN::GPInstructionC;

int testGeneInstIO()
{

  for(unsigned i = 0;i < 10000;i++) {
    RavlInfo("Test %d ",i);

    GenePaletteC::RefT palette = new GenePaletteC();
    palette->AddProxy("GPInstruction",*RavlN::GeneticN::InstructionMetaType());

    // Instantiate genome
    GeneFactoryC factory(*RavlN::GeneticN::InstructionMetaType(),*palette);
    GenomeC::RefT genome = &factory.Genome();

    GPInstructionC::RefT inst;
    factory.Get(inst);
    RAVL_TEST_TRUE(inst.IsValid());

    GenomeC::RefT genomeRL;
    if(!TestBinStreamIO(genome,genomeRL))
      return __LINE__;

    RAVL_TEST_EQUALS(genome->Size(),genomeRL->Size());
  }
  return 0;
}
