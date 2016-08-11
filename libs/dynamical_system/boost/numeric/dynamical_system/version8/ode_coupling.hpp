/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#  ifndef	_ADDITIVE_COUPLEABLE_
#  define   _ADDITIVE_COUPLEABLE_

#  include  "ode_helper.hpp"

//#include <utility>


namespace boost {
namespace numeric {
namespace dynamical_system {

struct linear_coupling
{
   double  weight;

   template < class state , class ftype >
   state  operator()  ( const state & x , const state & y , const ftype & t ) const {
      return  weight * y;
   }

};


struct diffusive_coupling
{
   double  weight;

   template < class state , class ftype >
   state  operator()  ( const state & x , const state & y , const ftype & t ) const {
      return  weight * ( y - x );
   }

};


#define    MAKE_COUPLING_WITH_1PARAM( NAME , PARAM1_NAME , PARAM1_VALUE , BODY )\
\
struct NAME \
{ \
   double  PARAM1_NAME; \
\
   NAME() : PARAM1_NAME ( PARAM1_VALUE ) { } \
\
   template < class state , class ftype > \
   state  operator() ( const state & x , const state & y , const ftype & t ) const \
   { \
      BODY \
   } \
};



#define    MAKE_COUPLING_WITH_2PARAM( NAME , PARAM1_NAME , PARAM1_VALUE , PARAM2_NAME , PARAM2_VALUE , BODY )\
\
struct NAME \
{ \
   double  PARAM1_NAME; \
   double  PARAM2_NAME; \
\
   NAME() : PARAM1_NAME ( PARAM1_VALUE ) , PARAM2_NAME ( PARAM2_VALUE ) { } \
\
   template < class state , class ftype > \
   state  operator() ( const state & x , const state & y , const ftype & t ) const \
   { \
      BODY \
   } \
};



};   // namespace dynamical_system
};   // namespace numeric
};   // namespace boost

#endif

