#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/VariableDiscrete.hh"

using namespace RavlProbN;
	
class RandomVariableDiscreteTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( RandomVariableDiscreteTest );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testNumValues );
	CPPUNIT_TEST( testValues );
	CPPUNIT_TEST( testValue );
	CPPUNIT_TEST_EXCEPTION( testValueThrows1, ExceptionC );
	CPPUNIT_TEST_EXCEPTION( testValueThrows2, ExceptionC );
	CPPUNIT_TEST( testIndex );
	CPPUNIT_TEST_EXCEPTION( testIndexThrows, ExceptionC );
	CPPUNIT_TEST_SUITE_END();
private:
	VariableDiscreteC m_variable;
public:
	void setUp() {
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variable = VariableDiscreteC("variable", names);
	}
	
	void tearDown() {
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_variable.ToString() == "Variable=<b,a,c>" );
	}
	
	void testNumValues() {
		CPPUNIT_ASSERT( m_variable.NumValues() == 3 );
	}
	
	void testValues() {
		CPPUNIT_ASSERT( m_variable.Values().Size() == 3 );
		CPPUNIT_ASSERT( m_variable.Values().Contains("a") == true );
		CPPUNIT_ASSERT( m_variable.Values().Contains("b") == true );
		CPPUNIT_ASSERT( m_variable.Values().Contains("c") == true );
		CPPUNIT_ASSERT( m_variable.Values().Contains("d") == false );
	}
	
	void testValue() {
		CPPUNIT_ASSERT( m_variable.Value(0) != m_variable.Value(1) );
		CPPUNIT_ASSERT( m_variable.Value(1) != m_variable.Value(2) );
	}
	
	void testValueThrows1() {
		m_variable.Value(3);
	}
	
	void testValueThrows2() {
		m_variable.Value(-1);
	}
	
	void testIndex() {
		CPPUNIT_ASSERT( m_variable.Value(m_variable.Index("a")) == "a" );
		CPPUNIT_ASSERT( m_variable.Value(m_variable.Index("b")) == "b" );
		CPPUNIT_ASSERT( m_variable.Value(m_variable.Index("c")) == "c" );
	}
	
	void testIndexThrows() {
		m_variable.Index("d");
	}
	
};

CPPUNIT_TEST_SUITE_REGISTRATION( RandomVariableDiscreteTest );
