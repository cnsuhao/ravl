#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/Proposition.hh"
#include "Ravl/Prob/RandomVariableValueBoolean.hh"
#include "Ravl/Prob/RandomVariableValueContinuous.hh"
#include "Ravl/Prob/RandomVariableValueDiscrete.hh"

using namespace RavlProbN;
	
class PropositionTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( PropositionTest );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testLotteryName );
	CPPUNIT_TEST( testDomain );
	CPPUNIT_TEST( testNumValues );
	CPPUNIT_TEST( testValues );
	CPPUNIT_TEST( testValue );
	CPPUNIT_TEST_EXCEPTION( testValueThrows1, ExceptionC );
	CPPUNIT_TEST_EXCEPTION( testValueThrows2, ExceptionC );
	CPPUNIT_TEST( testSubProposition );
	CPPUNIT_TEST_EXCEPTION( testSubPropositionThrows, ExceptionC );
	CPPUNIT_TEST( testEquality );
	CPPUNIT_TEST_SUITE_END();
private:
	DomainC m_domain;
	PropositionC m_proposition;
	HSetC<VariableC> m_variables;
	HSetC<VariablePropositionC> m_values;
public:
	void setUp() {
		VariableBooleanC booleanVariable("boolean");
		m_variables.Insert(booleanVariable);
		VariableContinuousC continuousVariable("continuous", RealRangeC(0.0, 1.0));
		m_variables.Insert(continuousVariable);
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variables.Insert(VariableDiscreteC("discrete", names));
		m_domain = DomainC(m_variables);
		m_values.Insert(RandomVariableValueBooleanC(booleanVariable, true));
		m_values.Insert(RandomVariableValueContinuousC(continuousVariable, 0.5));
		m_proposition = PropositionC(m_domain, m_values);
	}
	
	void tearDown() {
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_proposition.ToString() == "Continuous=0.500000,Boolean=boolean" );
	}
	
	void testLotteryName() {
		CPPUNIT_ASSERT( m_proposition.LotteryName() == "Boolean,Discrete,Continuous->(Discrete)" );
	}
	
	void testDomain() {
		CPPUNIT_ASSERT( m_proposition.Domain() == m_domain );
	}
	
	void testNumValues() {
		CPPUNIT_ASSERT( m_proposition.NumValues() == 2 );
	}
	
	void testValues() {
		CPPUNIT_ASSERT( m_proposition.Values().Size() == 2 );
		for (HSetIterC<VariablePropositionC> it(m_values); it; it++)
			CPPUNIT_ASSERT( m_proposition.Values().Contains(*it) == true );
	}
	
	void testValue() {
		CPPUNIT_ASSERT( m_proposition.Value(0) != m_proposition.Value(1) );
	}
	
	void testValueThrows1() {
		m_proposition.Value(2);
	}
	
	void testValueThrows2() {
		m_proposition.Value(-1);
	}
	void testSubProposition() {
		HSetC<VariableC> variables;
		variables.Insert(VariableBooleanC("boolean"));
		CPPUNIT_ASSERT( m_proposition.SubProposition(DomainC(variables)).NumValues() == 1 );
	}
	
	void testSubPropositionThrows() {
		HSetC<VariableC> variables;
		variables.Insert(VariableBooleanC("invalid"));
		m_proposition.SubProposition(DomainC(variables));
	}
	
	void testEquality() {
		PropositionC secondProposition(m_domain, m_values);
		HSetC<VariableC> variables;
		variables.Insert(VariableBooleanC("boolean"));
		PropositionC subProposition = m_proposition.SubProposition(DomainC(variables));
		CPPUNIT_ASSERT( m_proposition == m_proposition );
		CPPUNIT_ASSERT( m_proposition == secondProposition );
		CPPUNIT_ASSERT( !(m_proposition == subProposition) );
		CPPUNIT_ASSERT( m_proposition != subProposition );
		CPPUNIT_ASSERT( !(m_proposition != secondProposition) );
	}
		
};

CPPUNIT_TEST_SUITE_REGISTRATION( PropositionTest );
