#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_

#include <vector>
#include <tr1/array>
#include	"const_ode.hpp"
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
//      public  const_ode< _system::dimension * N_node >,
      public  const_network< _system , N_node >
   {
      using  const_network< _system , N_node > :: begin;
      using  const_network< _system , N_node > :: end;

      typedef     std::vector < double >        state;
//      typedef     std::tr1::array < double , _system::dimension * N_node >        state;

   private:

      class index2state {
         const typename state::const_iterator  s;
      public:
         typedef  tuple < typename state::const_iterator , double >  result_type;

         index2state ( const typename state::const_iterator _ ) : s(_) {}

         result_type  operator () ( const tuple< size_t , double >& _ ) const {
            return  result_type ( s + _system::dimension * get<0>(_) , get<1>(_) );
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

         index2state   i2s( x.begin() );
         typename state::const_iterator   i = x.begin();
         typename state::iterator         j = y.begin();

         const size_t dim = _system::dimension;
         for ( node* n = begin() ; n != end() ;  ++n , i += dim , j += dim) {
            (*n)( i , j , t , i2s_iter( n->in.begin(), i2s ) , i2s_iter( n->in.end(), i2s ) );
         }

      }

   };  // class const_dynamical_network

};     // namespace odeint
};     // namespace numeric
};     // namespace boost

#endif

