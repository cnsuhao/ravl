#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/VariableDiscrete.hh"

using namespace RavlProbN;
	
class VariableDiscreteTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( VariableDiscreteTest );
	CPPUNIT_TEST( testToString );
	CPPUNIT_TEST( testNumValues );
	CPPUNIT_TEST( testDomain );
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
		DomainDiscreteC domain;
		domain.Insert("a");
		domain.Insert("b");
		domain.Insert("c");
		m_variable = VariableDiscreteC("variable", domain);
	}
	
	void tearDown() {
	}
	
	void testToString() {
		CPPUNIT_ASSERT_EQUAL( StringC("Variable=<a,b,c>"), m_variable.ToString() );
	}
	
	void testNumValues() {
		CPPUNIT_ASSERT( m_variable.NumValues() == 3 );
	}
	
	void testDomain() {
		CPPUNIT_ASSERT( m_variable.Domain().Size() == 3 );
		CPPUNIT_ASSERT( m_variable.Domain().Contains("a") == true );
		CPPUNIT_ASSERT( m_variable.Domain().Contains("b") == true );
		CPPUNIT_ASSERT( m_variable.Domain().Contains("c") == true );
		CPPUNIT_ASSERT( m_variable.Domain().Contains("d") == false );
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

CPPUNIT_TEST_SUITE_REGISTRATION( VariableDiscreteTest );
