#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_

#include	"ode.hpp"
#include	"network.hpp"
#include <boost/iterator/transform_iterator.hpp>
#include <vector>

namespace boost {
namespace numeric {
namespace odeint {


   template <
      class   _system
   >
   struct ode_network : public  network< _system >
   {
      using  network< _system > :: nodee;

      typedef   std::vector < typename _system::state >   state;
      typedef  typename  network< _system > :: iterator   iterator;


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

      ode_network() { }
      ode_network( size_t N_node ) : network< _system > ( N_node ) { }


      void  operator() ( const state&  x , state&  y , const double  t )
      {
         typedef
            transform_iterator < index2state , list < tuple< size_t , double > > :: iterator >
               i2s_iter;

         typedef  typename network< _system >::node   node;

         index2state   i2s( x );
         typename state::const_iterator   i = x.begin();
         typename state::iterator         j = y.begin();

         for ( iterator n = nodee().begin() ; n != nodee().end() ; ++n )
            (*n)( *i++ , *j++ , t , i2s_iter( n->in.begin(), i2s ) , i2s_iter( n->in.end(), i2s ) );
      }

   };  // class const_dynamical_network

};     // namespace dynamical_system
};     // namespace boost
};     // namespace numeric

#endif

