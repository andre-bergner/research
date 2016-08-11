/*
 boost header: numeric/dynamical_system/ode.hpp

 Copyright 2010 André Bergner

 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_ode
#define		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_ode


#include	<boost/array.hpp>
#include	<boost/operators.hpp>


namespace boost {
namespace numeric {
namespace odeint {


   // the template class state_t defines the general behaviour of the state-type
   // used in dynamical_system classes
   template < class T , size_t N , class _T = double >
   struct state_t :
      public boost::array< T , N > ,
      boost::addable< state_t<T,N> ,
         boost::subtractable< state_t<T,N> ,
            boost::multipliable< state_t<T,N> , _T 
      > > >
   // TODO for nonlinear couplings we need element-wise multiplication as well
   {
      typedef  typename boost::array< T , N >::iterator         iterator;
      typedef  typename boost::array< T , N >::const_iterator   const_iterator;

      state_t<T,N>&  operator +=  ( const state_t<T,N>& s )
      {
         for ( size_t n=0 ; n < N ; ++n )  (*this)[n] += s[n];
         return *this;
      }

      state_t<T,N>&  operator -=  ( const state_t<T,N>& s )
      {
         for ( size_t n=0 ; n < N ; ++n )  (*this)[n] -= s[n];
         return *this;
      }

      template < class scalar >
      state_t<T,N>&  operator *=  ( scalar a )
      {
         for ( size_t n=0 ; n < N ; ++n )  (*this)[n] *= a;
         return *this;
      }
   };


   template <
      int    dim ,
      class  T = double
   >
   struct ode
   {
      const static int dimension = dim;	   // dimension of the whole system

      typedef  T                          base_type;
      typedef  state_t < T , dim , T >    state;

      void  operator()  (
               const state &  x,
                     state &  dxdt,
               const double   t
      ) const
      { }
   };

};   // namespace dynamical_system
};   // namespace numeric
};   // namespace boost


#define    MAKE_ODE( NAME , DIM , BODY )\
\
struct NAME : public ode< DIM > \
{ \
   void  operator() ( const state & x , state & dxdt , const double t ) const \
   { \
      BODY \
   } \
};

#define    MAKE_ODE_WITH_1PARAM( NAME , DIM , PARAM1_NAME , PARAM1_VALUE , BODY )\
\
class NAME : public ode< DIM > \
{ \
   double  PARAM1_NAME; \
public: \
   NAME() : PARAM1_NAME ( PARAM1_VALUE ) { } \
\
   void set_##PARAM1_NAME ( double _ ) { PARAM1_NAME = _; } \
\
   void  operator() ( const state & x , state & dxdt , const double t ) const \
   { \
      BODY \
   } \
};


#define    MAKE_ODE_WITH_2PARAM( \
              NAME , DIM , PARAM1_NAME , PARAM1_VALUE , PARAM2_NAME , PARAM2_VALUE , BODY )  \
\
class NAME : public ode< DIM > \
{ \
   double  PARAM1_NAME; \
   double  PARAM2_NAME; \
public: \
   NAME() : \
      PARAM1_NAME ( PARAM1_VALUE ), \
      PARAM2_NAME ( PARAM2_VALUE )  \
   { } \
\
   void set_##PARAM1_NAME ( double _ ) { PARAM1_NAME = _; } \
   void set_##PARAM2_NAME ( double _ ) { PARAM2_NAME = _; } \
\
   void  operator() ( const state & x , state & dxdt , const double t ) const \
   { \
      BODY \
   } \
};



#define    MAKE_ODE_WITH_3PARAM( NAME , DIM , PARAM1_NAME , PARAM1_VALUE ,\
              PARAM2_NAME , PARAM2_VALUE , PARAM3_NAME , PARAM3_VALUE , BODY )  \
\
class NAME : public ode< DIM > \
{ \
   double  PARAM1_NAME; \
   double  PARAM2_NAME; \
   double  PARAM3_NAME; \
public: \
   NAME() : \
      PARAM1_NAME ( PARAM1_VALUE ), \
      PARAM2_NAME ( PARAM2_VALUE ), \
      PARAM3_NAME ( PARAM3_VALUE )  \
   { } \
\
   void set_##PARAM1_NAME ( double _ ) { PARAM1_NAME = _; } \
   void set_##PARAM2_NAME ( double _ ) { PARAM2_NAME = _; } \
   void set_##PARAM3_NAME ( double _ ) { PARAM3_NAME = _; } \
\
   void  operator() ( const state & x , state & dxdt , const double t ) const \
   { \
      BODY \
   } \
};





#endif

