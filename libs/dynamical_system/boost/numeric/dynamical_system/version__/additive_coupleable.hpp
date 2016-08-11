#ifndef		_ADDITIVE_COUPLEABLE_
#define		_ADDITIVE_COUPLEABLE_

#include	"const_dynamical_system.hpp"

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

/*
#include <cmath>
template < size_t coupling_dim = 0 >
struct kuramoto_coupling {
   template < class state >
   state  operator() ( const state& x , const state& y , double weight ) {
      return  weight * sin( y[coupling_dim] - x[coupling_dim] );
   }
};
*/


template < class _system , class coupling_type = standard_coupling >
struct additive_coupleable : public _system
{
   coupling_type  coupling;

   // a redefinition of the original operator() which is hidden by the other (following) definition
   template < class const_state , class state , class time >
   void  operator()  ( const_state x, state dxdt, const time t ) const {
      _system :: operator() ( x , dxdt , t );
   }

   template < class const_state , class state , class time, class iterator >
   void  operator()  (
          const_state  x,
                state  dxdt,
           const time  t,
             iterator  n,        // FIXME must be a const_iterator -> how to formulate this ?
       const iterator  last_n )  // FIXME ditto
   {
      _system :: operator() ( x , dxdt , t );	// compute the state of the uncoupled system
      while ( n != last_n ) {
         for ( size_t d = 0 ; d < _system::dimension ; ++d )
            *dxdt++  += get<1>(*n) * get<0>(*n)[d];
         ++n;
      }
   }

};

};   // namespace odeint
};   // namespace numeric
};   // namespace boost

#endif

