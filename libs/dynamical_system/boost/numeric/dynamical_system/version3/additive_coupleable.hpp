#ifndef		_ADDITIVE_COUPLEABLE_
#define		_ADDITIVE_COUPLEABLE_

#include	"ode.hpp"

#include	<tr1/tuple>
using std::tr1::tuple;
using std::tr1::get;


namespace boost {
namespace numeric {
namespace odeint {


struct standard_coupling {
   template < class state >
   state  operator() ( const state& x , const state& y , double weight ) {
      return  weight * y;
   }
};


template < class _system , class coupling_type = standard_coupling >
struct additive_coupleable : public _system
{
   typedef   typename _system::state   state;

   coupling_type  coupling;

   // a redefinition of the original operator() which is hidden by the other (following) definition
   void  operator()  ( const state& x , state& dxdt , const double t ) {
      _system :: operator() ( x , dxdt , t );
   }

   template < class iterator >
   void  operator()  (
         const state&  x,
               state&  dxdt,
         const double  t,
             iterator  n,        // FIXME must be a const_iterator -> how to formulate this ?
       const iterator  last_n )  // FIXME ditto
   {
      _system :: operator() ( x , dxdt , t );	// compute the state of the uncoupled system
      while ( n != last_n ) {
//         dxdt  +=  get<1>(*n) * get<0>(*n);    // FIXME the coupling constant should be matrix !!!
         dxdt  +=  coupling( x , get<0>(*n) , get<1>(*n) );
         ++n;
      }
   }

};

};   // namespace dynamical_system
};   // namespace numeric
};   // namespace boost

#endif

