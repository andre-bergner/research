/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */


#  include  <vector>
#  include  <boost/iterator/transform_iterator.hpp>





namespace boost {


template <
   VertexProperty,
   EdgeProperty,
   VertexContainer = std::vector,
   EdgeContainer   = std::vector
>
struct ez_adjacency_list {


   ez_adjacency_list() {}

   ez_adjacency_list( size_t  N_vertex )
   :  _vertex_list ( N_vertex )
   {}


private:


   typedef  EdgeProperty   _EdgeInformation;


   struct _VertexInformation {
      VertexProperty                      _property;
      EdgeContainer< _EdgeInformation >   _in_edge_list;
   }


   VertexContainer< Vertex >     _vertex_list;


};





}

