# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2011, Charles Galambos
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! author=Charles Galambos
#! docentry=Ravl.API.Math.Genetic.Optimisation

PACKAGE=Ravl/Genetic

HEADERS=Genome.hh GeneticOptimiser.hh \
 GenomeConst.hh GenomeShare.hh GenomeList.hh GenomeClass.hh GenomeMeta.hh \
 GeneTypeWeightedMeta.hh GeneTypeFloatGauss.hh \
 EvaluateFitness.hh EvaluateFitnessFunc.hh

SOURCES=Genome.cc GeneticOptimiser.cc \
 GenomeConst.cc GenomeShare.cc GenomeList.cc GenomeClass.cc GenomeMeta.cc \
 GeneTypeWeightedMeta.cc GeneTypeFloatGauss.cc \
 Gene2ClassGenerator.cc \
 EvaluateFitness.cc EvaluateFitnessFunc.cc

MAINS= 

EXAMPLES=exGeneticOptimisation.cc

MUSTLINK=LinkGeneticOpt.cc

PLIB=RavlGeneticOptimisation

USESLIBS=RavlOS RavlIO RavlXMLFactory

PROGLIBS= RavlGeneticProgram 

AUXFILES= exGeneticOptimisation.xml

AUXDIR= /share/Ravl/Genetic
 
TESTEXES=testGeneticOpt.cc
