/////////////////////////////////////////////////////////////////////
//
//  class namix::additive_coupleable
//
//
/////////////////////////////////////////////////////////////////////

#ifndef		_ADDITIVE_COUPLEABLE_
#define		_ADDITIVE_COUPLEABLE_


#include	"../const_dynamical_system.hpp"


namespace boost {
namespace numeric {
namespace dynamical_system {


template < class _system >
class additive_coupleable : public _system
{
//    using  typename _system::state;	// worx not with gcc 4.2 but with icc 11

public:

    typedef   typename _system::state   state;


    void  operator()  (
        const state &  x,
              state &  dxdt,
        const double   t )
    {
        _system :: operator() ( x , dxdt , t );
    }


    // TODO ext. forcing must not be of type state
    void  operator()  (
        const state &  x,
              state &  dxdt,
        const double   t,
        const state &  ext )
    {
        _system :: operator() ( x , dxdt , t );	// compute the state of the system
        for ( int d = 0 ; d < _system::dimension ; ++d )
            dxdt[d]  +=  ext[d];		            // add the provided external force
    }


    // TODO: interchange loops( dim , nodes ) -> compare speed
    template<
        class  net_state,
        class  _C_ >
    static state &   coupling_function (
        net_state   x,
        int         n_node,
        _C_         beg,
        _C_         end,
        double      global_coupling_strength = 1.0 )
    {
        static  typename _system::state  tmp;
        for ( int d = 0  ;  d < _system::dimension  ; ++d )   tmp[d]  =  0.0;
        while ( beg != end ) {
            for ( int d = 0  ;  d < _system::dimension  ; ++d )   tmp[d] += beg->weight * x[beg->id][d];
//            for ( int d = 0  ;  d < _system::dimension  ; ++d )   tmp[d] += beg->weight * x(beg->id,d);
            ++beg;
        }
        for ( int d = 0  ;  d < _system::dimension  ; ++d )   tmp[d]  *= global_coupling_strength;
        return tmp;
    }


};   // namespace dynamical_system
};   // namespace boost
};   // namespace numeric

};  // namespace namix

#endif

