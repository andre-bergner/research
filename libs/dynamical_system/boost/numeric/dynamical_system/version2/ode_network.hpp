#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_

#include	"ode.hpp"
#include	"const_network.hpp"
#include <boost/iterator/transform_iterator.hpp>

namespace boost {
namespace numeric {
namespace odeint {


   template <
      class   _system,
      size_t  N_node
   >
   struct ode_network :
      public  ode< _system::dimension * N_node , typename _system::base_type >,
      public  const_network< _system , N_node >
   {
      using  const_network< _system , N_node > :: begin;
      using  const_network< _system , N_node > :: end;

      typedef
         state_t< typename _system::state , N_node , typename _system::base_type >
            state;

   private:

      class index2state {
         const state&  s;
      public:
         typedef  tuple < typename _system::state , double >  result_type;

         index2state ( const state& _ ) : s(_) {}

         result_type  operator () ( const tuple< size_t , double >& _ ) const {
            return  result_type ( s[ get<0>(_) ] , get<1>(_) );
         }
      };

   public:

      void  operator() ( const state&  x , state&  y , const double  t )
      {
         typedef
            transform_iterator < index2state , list < tuple< size_t , double > > :: iterator >
               i2s_iter;

         typedef
            typename const_network< _system , N_node >::node
               node;

         index2state   i2s( x );
         typename state::const_iterator   i = x.begin();
         typename state::iterator         j = y.begin();

         for ( node* n = begin() ; n != end() ; ++n )
            (*n)( *i++ , *j++ , t , i2s_iter( n->in.begin(), i2s ) , i2s_iter( n->in.end(), i2s ) );
      }

   };  // class const_dynamical_network

};     // namespace dynamical_system
};     // namespace boost
};     // namespace numeric

#endif

