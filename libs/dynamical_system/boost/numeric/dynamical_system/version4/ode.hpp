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


   template <  class L, class Op, class R >
   struct Xpression {
      const  L& l;
      const  R& r;

      Xpression ( const L& l_, const R& r_) :  l ( l_ ) , r ( r_ )  {  }

      double operator[] ( size_t n ) const {
         return  Op::_( l[n] , r[n] );
      }
   };

   template < class Op, class R >
   struct Xpression<double,Op,R> {
      const  double& l;
      const  R& r;

      Xpression ( const double& l_, const R& r_) :  l ( l_ ) , r ( r_ )  {  }

      double operator[] ( size_t n ) const {
         return  Op::_( l , r[n] );
      }
   };

   struct Times { static double _( double a , double b )  {  return  a * b;  } };
   struct Plus  { static double _( double a , double b )  {  return  a + b;  } };


   template < class L , class R >
   Xpression < L , Plus , R >   operator +  ( const L& l , const R &r ) {
      return  Xpression < L , Plus , R >  ( l , r );
   }

   template < class L , class R >
   Xpression < L , Times , R >   operator *  ( const L& l , const R &r ) {
      return  Xpression < L , Times , R >  ( l , r );
   }



   template < class T , size_t N >
   struct state_t : public boost::array< T , N > {

      state_t () {}

      template < typename L , typename Op , typename R >
      state_t ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }

      template < typename L , typename Op , typename R >
      void  operator =  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }

      template < typename L , typename Op , typename R >
      void  operator +=  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] += x[n]; }

      template < typename L , typename Op , typename R >
      void  operator -=  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] -= x[n]; }

      void  operator =  ( const state_t & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }

      void  operator +=  ( const state_t & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] += x[n];  }

      void  operator -=  ( const state_t & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] -= x[n];  }



   };






   ////////////////////////////////////////////////////////////////////
  ///
 //

   template <
      int    dim ,
      class  T = double
   >
   struct ode
   {
      const static int dimension = dim;	   // dimension of the whole system

      typedef  T                      base_type;
      typedef  state_t < T , dim >    state;

      void  operator()  (
               const state &  x,
                     state &  dxdt,
               const double   t
      ) const
      { }
   };

};   // namespace odeint
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

