/*
 boost header: numeric/dynamical_system/const_dynamical_system.hpp

 Copyright 2010 André Bergner

 Distributed under the Boost Software License, Version 1.0.
 (See accompanying file LICENSE_1_0.txt or
 copy at http://www.boost.org/LICENSE_1_0.txt)
*/

#ifndef		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_CONST_DYNAMICAL_SYSTEM
#define		BOOST_NUMERIC_DYNAMICAL_SYSTEMS_CONST_DYNAMICAL_SYSTEM


namespace boost {
namespace numeric {
namespace odeint {


   template < size_t dim >
   struct const_ode
   {
      const static int dimension = dim;	   // dimension of the whole system

      template <
         class const_state,
         class state,
         class time
      >
      void  operator()  (
               const const_state&  x,
                     state&  dxdt,
               const time    t
      ) const
      { }
   };

};   // namespace odeint
};   // namespace numeric
};   // namespace boost


#define    MAKE_ODE( NAME , DIM , BODY )\
\
struct NAME : public const_ode< DIM > \
{ \
   template < class const_state , class state, class time > \
   void  operator() ( const const_state & x , state & dxdt , const time t ) const \
   { \
      BODY \
   } \
};

#define    MAKE_ODE_WITH_1PARAM( NAME , DIM , PARAM1_NAME , PARAM1_VALUE , BODY )\
\
class NAME : public const_ode< DIM > \
{ \
   double  PARAM1_NAME; \
public: \
   NAME() : PARAM1_NAME ( PARAM1_VALUE ) { } \
\
   void set_##PARAM1_NAME ( double _ ) { PARAM1_NAME = _; } \
\
   template < class const_state , class state, class time > \
   void  operator() ( const const_state & x , state & dxdt , const time t ) const \
   { \
      BODY \
   } \
};


#define    MAKE_ODE_WITH_2PARAM( \
              NAME , DIM , PARAM1_NAME , PARAM1_VALUE , PARAM2_NAME , PARAM2_VALUE , BODY )  \
\
class NAME : public const_ode< DIM > \
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
   template < class const_state , class state, class time > \
   void  operator() ( const const_state & x , state & dxdt , const time t ) const \
   { \
      BODY \
   } \
};



#define    MAKE_ODE_WITH_3PARAM( NAME , DIM , PARAM1_NAME , PARAM1_VALUE ,\
              PARAM2_NAME , PARAM2_VALUE , PARAM3_NAME , PARAM3_VALUE , BODY )  \
\
class NAME : public const_ode< DIM > \
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
   template < class const_state , class state, class time > \
   void  operator() ( const const_state & x , state & dxdt , const time t ) const \
   { \
      BODY \
   } \
};


#endif

