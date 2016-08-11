#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_

#include	"ode.hpp"

#include <boost/iterator/transform_iterator.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>

#include <vector>
#include <utility>

namespace boost {
namespace numeric {
namespace odeint {


  /* TODO

   * test changing node number while running, espcecially focus on how indeces might change
     -> maybe use index-map to resolve problems if occuring

   * add possibility to use adjacency_matrix instead of list

   */


   struct  __link  {  double  weight;  };    // definition of the link property


   template <
      class   system
     ,class   link = __link      // TODO make this changeable, for now use __list
   >
   struct  adjacency {
      typedef  boost::adjacency_list <
                  boost::listS,           // use std::list as node container
                  boost::vecS,            // use std::vector as link container
                  boost::bidirectionalS,  // this is a directed graph
                  system,                 // the node property
                  link >                  // the link property
               list;

      // TODO dows not work currently, probably we have to use graph_traits
      typedef  boost::adjacency_matrix <
                  boost::bidirectionalS,  // this is a directed graph
                  system,                 // the node property
                  link >                  // the link property
               matrix;
   };


   template <
      class   system
     ,class   sys_state = typename system::state
   >
   struct ode_network : public  adjacency < system > :: list
   {
      typedef   typename adjacency < system > :: list   network;
      typedef   std::vector < sys_state >               state;

   private:

      class index2state {
         const  network &   n;
         const  state &     s;
      public:
         typedef  std::pair < sys_state , double >  result_type;

         index2state ( const network &  _n , const state &  _s ) : n(_n) , s(_s) {}

         result_type  operator () ( const typename network::edge_descriptor & _ ) const {
            return  result_type ( s[ source(_,n) ] , n[_].weight );
         }
      };

   public:

      ode_network() { }
      ode_network( size_t N_node ) : network ( N_node ) { }

      /* TODO converting constructor that copies the graph topology
      template < class system2 >
      ode_network( const adjacency < system2 > :: list & net )
         : network ( (network)net ) { }
      */

      void  operator() ( const state&  x , state&  y , const double  t )
      {

         typedef
            transform_iterator < index2state , typename network::in_edge_iterator >
               i2s_iter;

         index2state   i2s( *this , x );

         typename state::const_iterator   i = x.begin();
         typename state::iterator         j = y.begin();

         typename  network::vertex_iterator  node;
         for ( node  = vertices(*this).first;  node != vertices(*this).second;  ++node )
            (*this)[*node]( *i++ , *j++ , t ,
                            i2s_iter( in_edges(*node,*this).first, i2s ) ,
                            i2s_iter( in_edges(*node,*this).second, i2s )
                          );

      }

   };  // class ode_network


   ////////////////////////////////////////////////////////////////////////////
   //
   //  some helpet functions for easy network access and standard topologies
   //  TODO: move into own header


   template <
      class  graph,
      class  vertex
   >
   void add_link ( graph & g , vertex source , vertex target , double weight ) {
      g [ add_edge ( source , target , g ).first ].weight =  weight;
   }


   template <
      class  graph,
      class  vertex
   >
   typename graph::edge_bundled&  add_link ( graph & g , vertex source , vertex target ) {
      return  g [ add_edge ( source , target , g ).first ];
   }


   /////////////////////////////////////

   template <
      class  graph,
      class  vertex
   >
   typename graph::edge_bundled&  add_link ( graph & g , std::pair<vertex,vertex> vv ) {
      return  g [ add_edge ( vv.first , vv.second , g ).first ];
   }

   // wrapper class
   class _node {
      size_t  i;
   public:
      _node ( size_t _ ) : i(_) { }
      operator size_t () { return i; }
   };

   // connector operator
   std::pair<_node,_node>  operator <= ( const _node& _1 , const _node& _2 ) {
      return  std::make_pair( _2 , _1 );
   }


   ////////////////////////////////////////
/*
   template <
      class  graph,
      class  vertex
   >
   typename graph::vertex_descriptor&  add_links (
      graph &                                                   g ,
      graph::vertex_descriptor &                                v ,
      std::pair<graph::vertex_iteraor,graph::vertex_iteraor> &  l  )
   {
      foreach ( l.first , l.second , add_link ( graph & g , vertex source , vertex target );
      return  g [ add_edge ( source , target , g ).first ];
   }
*/






};     // namespace odeint
};     // namespace numeric
};     // namespace boost

#endif

