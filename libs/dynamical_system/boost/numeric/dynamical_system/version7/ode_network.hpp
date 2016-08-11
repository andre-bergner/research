/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_

#include	"ode.hpp"

#include <boost/iterator/transform_iterator.hpp>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/copy.hpp>

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


   template <
      class   system
     ,class   link
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
     ,class   coupling
     ,class   sys_state = typename system::state
   >
   struct ode_network : public  adjacency < system , coupling > :: list
   {
      typedef   typename adjacency < system , coupling > :: list   network;
      typedef   std::vector < sys_state  >                         state;

   public:

      ode_network() { }
      ode_network( size_t N_node ) : network ( N_node ) { }

      /* TODO converting constructor that copies the graph topology */
      template < class s , class c >
      ode_network( const typename adjacency<s,c>::list & net )
         //: network ( (network)net )
      { }

      template < class s , class c , class t>
      ode_network( const ode_network<s,c,t> & net )
      { }

//      template < class s , class c >
//      ode_network( const network & net )
//         : network ( (network)net ) { }


      void  operator() ( const state&  x , state&  dxdt , const double  t )
      {

         typename  state::const_iterator     i = x.begin();
         typename  state::iterator           j = dxdt.begin();
         typename  network::vertex_iterator  node;

         for ( node  = vertices(*this).first;  node != vertices(*this).second;  ++node ) {

            (*this)[*node]( *i , *j , t );
//            (*this)[*node]( x[*node] , dxdt[*node] , t );

            typename network::in_edge_iterator
               n0 = in_edges ( *node , *this ). first,
               n1 = in_edges ( *node , *this ). second;
            while ( n0 != n1 ) {
//               dxdt[*node]  +=  (*this)[*n0]( x[*node] , x[ source(*n0,*this) ] , t );
               dxdt[*node]  +=  (*this)[*n0]( *i , x[ source(*n0,*this) ] , t );
               ++n0;
            }

            ++i , ++j;
         }

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



   struct vertex_wrapper { const size_t index; };

   auto vtx(size_t n) { return vertex_wrapper{n}; }

   // connector operator
   std::pair<size_t,size_t>  operator <= ( vertex_wrapper target , vertex_wrapper source )
   {
      return  std::make_pair( source.index , target.index );
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

