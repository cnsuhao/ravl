#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/VariablePropositionDiscrete.hh"

using namespace RavlProbN;
	
class VariablePropositionDiscreteTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( VariablePropositionDiscreteTest );
	CPPUNIT_TEST_EXCEPTION( testCreateThrows, ExceptionC );
	CPPUNIT_TEST( testStringValue );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testIndex );
	CPPUNIT_TEST( testEquality );
	CPPUNIT_TEST_SUITE_END();
private:
	VariableDiscreteC m_variable;
	VariablePropositionDiscreteC m_valueA;
	VariablePropositionDiscreteC m_valueB;
public:
	void setUp() {
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variable = VariableDiscreteC("variable", names);
		m_valueA = VariablePropositionDiscreteC(m_variable, "a");
		m_valueB = VariablePropositionDiscreteC(m_variable, "b");
	}
	
	void tearDown() {
	}
	
	void testCreateThrows() {
		VariablePropositionDiscreteC v(m_variable, "d");
	}
	
	void testStringValue() {
		CPPUNIT_ASSERT( m_valueA.Value() == "a" );
		CPPUNIT_ASSERT( m_valueB.Value() == "b" );
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_valueA.ToString() == "a" );
		CPPUNIT_ASSERT( m_valueB.ToString() == "b" );
	}
	
	void testIndex() {
		CPPUNIT_ASSERT( m_variable.Value(m_valueA.Index()) == m_valueA.Value());
		CPPUNIT_ASSERT( m_variable.Value(m_valueB.Index()) == m_valueB.Value());
	}
	
	void testEquality() {
		VariablePropositionDiscreteC secondValueA(m_variable, "a");
		CPPUNIT_ASSERT( m_valueA == m_valueA );
		CPPUNIT_ASSERT( m_valueA == secondValueA );
		CPPUNIT_ASSERT( !(m_valueA == m_valueB) );
		CPPUNIT_ASSERT( m_valueA != m_valueB );
		CPPUNIT_ASSERT( !(m_valueA != m_valueA) );
		CPPUNIT_ASSERT( !(m_valueA != secondValueA) );
	}
};

CPPUNIT_TEST_SUITE_REGISTRATION( VariablePropositionDiscreteTest );
