#include <cppunit/extensions/HelperMacros.h>

#include "Ravl/Prob/Domain.hh"
#include "Ravl/Prob/RandomVariableBoolean.hh"
#include "Ravl/Prob/RandomVariableContinuous.hh"
#include "Ravl/Prob/VariableDiscrete.hh"

using namespace RavlProbN;
	
class DomainTest: public CppUnit::TestCase {
	CPPUNIT_TEST_SUITE( DomainTest );
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
	DomainC m_domain;
	HSetC<VariableC> m_variables;
public:
	void setUp() {
		m_variables.Insert(RandomVariableBooleanC("boolean"));
		m_variables.Insert(RandomVariableContinuousC("continuous", RealRangeC(0.0, 1.0)));
		HSetC<StringC> names;
		names.Insert("a");
		names.Insert("b");
		names.Insert("c");
		m_variables.Insert(VariableDiscreteC("discrete", names));
		m_domain = DomainC(m_variables);
	}
	
	void tearDown() {
	}
	
	void testEquality() {
		DomainC secondDomain(m_variables);
		HSetC<VariableC> emptySet;
		DomainC thirdDomain(emptySet);
		CPPUNIT_ASSERT( m_domain == m_domain );
		CPPUNIT_ASSERT( m_domain == secondDomain );
		CPPUNIT_ASSERT( !(m_domain == thirdDomain) );
		CPPUNIT_ASSERT( m_domain != thirdDomain );
		CPPUNIT_ASSERT( !(m_domain != secondDomain) );
	}
	
	void testContains() {
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_domain.Contains(*it) == true );
		CPPUNIT_ASSERT( m_domain.Contains(VariableC()) == false);
		CPPUNIT_ASSERT( m_domain.Contains(RandomVariableBooleanC("bool2")) == false );
	}
	
	void testNumVariables() {
		CPPUNIT_ASSERT( m_domain.NumVariables() == 3 );
	}
	
	void testVariables() {
		CPPUNIT_ASSERT( m_domain.Variables().Size() == 3 );
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_domain.Variables().Contains(*it) == true );
	}
	
	void testVariable() {
		CPPUNIT_ASSERT( m_domain.Variable(0) != m_domain.Variable(1) );
		CPPUNIT_ASSERT( m_domain.Variable(1) != m_domain.Variable(2) );
	}
	
	void testVariableThrows1() {
		m_domain.Variable(3);
	}
	
	void testVariableThrows2() {
		m_domain.Variable(-1);
	}
	
	void testIndex() {
		for (HSetIterC<VariableC> it(m_variables); it; it++)
			CPPUNIT_ASSERT( m_domain.Variable(m_domain.Index(*it)) == *it );
	}
	
	void testIndexThrows() {
		m_domain.Index(VariableC());
		m_domain.Index(RandomVariableBooleanC("bool3"));
	}
	
	void testToString() {
		CPPUNIT_ASSERT( m_domain.ToString() == "Boolean,Discrete,Continuous" );
	}
	
};

CPPUNIT_TEST_SUITE_REGISTRATION( DomainTest );
