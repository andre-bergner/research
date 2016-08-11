/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#  ifndef   BOOST_NUMERIC_DYNAMICAL_SYSTEMS_ODE_HELPER
#  define   BOOST_NUMERIC_DYNAMICAL_SYSTEMS_ODE_HELPER


//#  include   <boost/array.hpp>
#  include  <array>
#  include  <boost/operators.hpp>
#  include  <type_traits>


namespace boost {
namespace numeric {
namespace dynamical_system {


/** VERSION WITH EXPRESSION TEMPLATES **************************************/




   template < bool >
   struct index_trait {
      template < class T , class I >
      static  double index ( const T & t , const I & i )   { return  t[i]; }
   };

   template <>
   struct index_trait<true> {
      template < class T , class I >
      static  T index ( const T & t , const I & )  { return  t; }
   };

   template <  class L, class Op, class R >
   struct Xpression {
      const  L  & l_;
      const  R  & r_;
      
      Xpression ( const L & l, const R & r ) :  l_(l) , r_(r)  { }
      
      double  operator[] ( size_t n ) const {
         return  Op::apply(
                    index_trait< std::is_fundamental<L>::value >::index( l_ , n ),
                    index_trait< std::is_fundamental<R>::value >::index( r_ , n )
                 );
      }
   };


   struct __Times {
      template<class T , class S>
      static double apply( const T & t , const S & s )  { return  t * s; }
   };

   struct __Minus {
      template<class T , class S>
      static double apply( const T & t , const S & s )  { return  t - s; }
   };

   struct __Plus {
      template<class T , class S>
      static double apply( const T & t , const S & s )  { return  t + s; }
   };
   


   template < class L , class R >
   Xpression < L , __Plus , R >   operator +  ( const L& l , const R &r ) {
      return  Xpression < L , __Plus , R >  ( l , r );
   }

   template < class L , class R >
   Xpression < L , __Minus , R >   operator -  ( const L& l , const R &r ) {
      return  Xpression < L , __Minus , R >  ( l , r );
   }

   template < class L , class R >
   Xpression < L , __Times , R >   operator *  ( const L& l , const R &r ) {
      return  Xpression < L , __Times , R >  ( l , r );
   }



   template <
      class     T
    , unsigned  N
   >
   struct state_t : public std::array< T , N > {

      state_t () {}


      template < typename L , typename Op , typename R >
      state_t ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }
      

      /*
       *          operators accepting expressions
       */

      template < typename L , typename Op , typename R >
      void  operator =  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }

      template < typename L , typename Op , typename R >
      void  operator +=  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] += x[n]; }

      template < typename L , typename Op , typename R >
      void  operator -=  ( const Xpression<L,Op,R> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] -= x[n]; }


      /*
       *          operators accepting state_t types
       */

      template < class U >
      void  operator =  ( const state_t<U,N> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = x[n];  }

      template < class U  >
      void  operator +=  ( const state_t<U,N> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] += x[n];  }

      template < class U  >
      void  operator -=  ( const state_t<U,N> & x ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] -= x[n];  }


      /*
       *          scalar operations
       */

      template < class U >
      void  operator =  ( const U & a ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] = a;  }

      template < class U >
      void  operator -=  ( const U & a ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] -= a;  }

      template < class U >
      void  operator +=  ( const U & a ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] += a;  }

      template < class U >
      void  operator *=  ( const U & a ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] *= a;  }
      
      template < class U >
      void  operator /=  ( const U & a ) {
         for ( size_t n = 0; n < N ; ++n )   (*this)[n] /= a;  }
      
   };



   ////////////////////////////////////////////////////////////////////


   template <
      int    dim ,
      class  T = double
   >
   struct static_ode
   {
      const static int dimension = dim;	   // dimension of the whole system

      typedef  T                      base_type;
      typedef  state_t < T , dim >    state_type;


//      state_type   _state;

//      state_type&  state() { return _state; }

      
      void  operator()  (
               const state_type &  x,
                     state_type &  dxdt,
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
   void  operator() ( const state_type & x , state_type & dxdt , const double t ) const \
   { \
      BODY \
   } \
};

#define    MAKE_ODE_WITH_1PARAM( NAME , DIM , PARAM1_NAME , PARAM1_VALUE , BODY )\
\
class NAME : public ode< DIM > \
{ \
public: \
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
public: \
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
public: \
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

