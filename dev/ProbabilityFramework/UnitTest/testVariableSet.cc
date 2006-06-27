#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/VariableSet.hh"
#include "Ravl/Prob/VariableBoolean.hh"
#include "Ravl/Prob/VariableContinuous.hh"
#include "Ravl/Prob/VariableDiscrete.hh"

using namespace RavlProbN;
	
class VariableSetTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( VariableSetTest );
	CPPUNIT_TEST( testEquality );
	CPPUNIT_TEST( testContains );
	CPPUNIT_TEST( testNumVariables );
	CPPUNIT_TEST( testVariables );
	CPPUNIT_TEST( testVariable );
	CPPUNIT_TEST_EXCEPTION( testVariableThrows1, ExceptionC );
	CPPUNIT_TEST_EXCEPTION( testVariableThrows2, ExceptionC );
	CPPUNIT_TEST( testIndex );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST_EXCEPTION( testIndexThrows, ExceptionC );
	CPPUNIT_TEST_SUITE_END();
private:
	VariableSetC m_variableSet;
	HSetC<VariableC> m_variables;
public:
	void setUp() {
		m_variables.Insert(VariableBooleanC("boolean"));
		m_variables.Insert(VariableContinuousC("continuous", RealRangeC(0.0, 1.0)));
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variables.Insert(VariableDiscreteC("discrete", names));
		m_variableSet = VariableSetC(m_variables);
	}
	
	void tearDown() {
	}
	
	void testEquality() {
		VariableSetC secondVariableSet(m_variables);
		HSetC<VariableC> emptySet;
		VariableSetC thirdVariableSet(emptySet);
		CPPUNIT_ASSERT( m_variableSet == m_variableSet );
		CPPUNIT_ASSERT( m_variableSet == secondVariableSet );
		CPPUNIT_ASSERT( !(m_variableSet == thirdVariableSet) );
		CPPUNIT_ASSERT( m_variableSet != thirdVariableSet );
		CPPUNIT_ASSERT( !(m_variableSet != secondVariableSet) );
	}
	
	void testContains() {
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_variableSet.Contains(*it) == true );
		CPPUNIT_ASSERT( m_variableSet.Contains(VariableC()) == false);
		CPPUNIT_ASSERT( m_variableSet.Contains(VariableBooleanC("bool2")) == false );
	}
	
	void testNumVariables() {
		CPPUNIT_ASSERT( m_variableSet.NumVariables() == 3 );
	}
	
	void testVariables() {
		CPPUNIT_ASSERT( m_variableSet.Variables().Size() == 3 );
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_variableSet.Variables().Contains(*it) == true );
	}
	
	void testVariable() {
		CPPUNIT_ASSERT( m_variableSet.Variable(0) != m_variableSet.Variable(1) );
		CPPUNIT_ASSERT( m_variableSet.Variable(1) != m_variableSet.Variable(2) );
	}
	
	void testVariableThrows1() {
		m_variableSet.Variable(3);
	}
	
	void testVariableThrows2() {
		m_variableSet.Variable(-1);
	}
	
	void testIndex() {
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_variableSet.Variable(m_variableSet.Index(*it)) == *it );
	}
	
	void testIndexThrows() {
		m_variableSet.Index(VariableC());
		m_variableSet.Index(VariableBooleanC("bool3"));
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_variableSet.ToString() == "Boolean,Discrete,Continuous" );
	}
	
};

CPPUNIT_TEST_SUITE_REGISTRATION( VariableSetTest );
