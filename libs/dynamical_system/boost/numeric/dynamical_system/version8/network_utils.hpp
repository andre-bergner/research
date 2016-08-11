/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef		_BOOST_NUMERIC_DYNAMICAL_SYSTEM_NETWORK_UTILS_
#define		_BOOST_NUMERIC_DYNAMICAL_SYSTEM_NETWORK_UTILS_


namespace boost {
namespace numeric {              // TODO
namespace dynamical_system {     // -> what namespace to put this stuff in ?


   ////////////////////////////////////////////////////////////////////////////
   //
   //  some helper functions for easy network access and standard topologies
   //  TODO: move into own header
   //
   //


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


   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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



   // - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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



};     // namespace dynamical_system
};     // namespace numeric
};     // namespace boost

#endif

