/*
 *  boost header: numeric/dynamical_system/ode_helper.hpp
 *
 *  Copyright 2010-2011 André Bergner
 *
 *  Distributed under the Boost Software License, Version 1.0.
 *  (See accompanying file LICENSE_1_0.txt or
 *  copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef		_BOOST_NUMERIC_DYNAMICAL_SYSTEM_ODE_NETWORK_
#define		_BOOST_NUMERIC_DYNAMICAL_SYSTEM_ODE_NETWORK_

#include	"ode_helper.hpp"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/adjacency_matrix.hpp>
#include <boost/graph/copy.hpp>

#include <boost/iterator/transform_iterator.hpp>
#include <boost/ref.hpp>

#include <functional>
#include <vector>
#include <list>
#include <utility>


//#include <boost/numeric/odeint/util/state_wrapper.hpp>

/*
 *
 *    TODO
 *
 *    • StateProxy iterators are not default constructable --> check and fix if possible
 *
 *
 *
 *
 *
 *
 */

namespace boost {
namespace numeric {
namespace dynamical_system {


  /* TODO

   * add possibility to use adjacency_matrix instead of list

   */

   namespace ode_network_detail {

      template <
         class   vertex_prop
        ,class   edge_prop
      >
      struct  adjacency {
         typedef  boost::adjacency_list <
                     boost::listS,            // ...
                     boost::vecS,             // this must be vector due to auto indexing !!!
                     boost::bidirectionalS,   // this is a directed graph with in_edge_iterators
                     vertex_prop,             // the node property
                     edge_prop >              // the edge property
                  list;

         // TODO dows not work currently, probably we have to use graph_traits
         typedef  boost::adjacency_matrix <
                     boost::bidirectionalS,   // this is a directed graph
                     vertex_prop,             // the node property
                     edge_prop >              // the edge property
                  matrix;
      };



      template <
         class System ,
         class State
      >
      struct system_decorator : public System {
//      ~system_wrapper() { container.erase( iter_state ); }
//       Container   container;     // redundant - exists for every node
//       Iterator    iter_state;
         State    state;
      };




      //  this class is responsible for wrapping the network class
      //  and make it look like an state type from the outside
      //
      template < class Network >
      class StateProxy {

         typedef    typename Network::sys_state_type   sys_state_type;

         Network &   _net;
         
         
         class node2state {         // the transfomer class which transforms
            Network &   _net;

         public:
            typedef   sys_state_type &  result_type;
            
            node2state ( Network & net ) : _net( net )  { }
            
            result_type  operator () ( const typename graph_traits<Network>::vertex_descriptor & v ) const {
               return   _net[v].state;
            }
         };
         
         
         class const_node2state {         // the transfomer class which transforms
            const Network &   _net;
         public:
            typedef   const sys_state_type &  result_type;
            
            const_node2state ( const Network & net ) : _net( net )  { }
            
            result_type  operator () ( const typename graph_traits<Network>::vertex_descriptor & v ) const {
               return   _net[v].state;
            }
         };
         
         
         node2state  _n2s;
         const_node2state  _n2s_c;
         
      public:
         
         StateProxy ( Network & net )
         :  _net ( net ),
            _n2s ( net ),
            _n2s_c ( net )
         { }
         
         typedef
            transform_iterator< node2state , typename graph_traits<Network>::vertex_iterator >
               StateIterator;
         
         typedef
            transform_iterator< const_node2state , typename graph_traits<Network>::vertex_iterator >
               const_StateIterator;
         
         typedef   StateIterator         iterator;
         typedef   const_StateIterator   const_iterator;
         
         
         const_StateIterator  begin () const {
            return const_StateIterator ( vertices(_net).first , _n2s_c  );
         }
         
         StateIterator  begin () {
            return StateIterator ( vertices(_net).first , _n2s  );
         }
         
         
         const_StateIterator  end () const {
            return const_StateIterator ( vertices(_net).second , _n2s_c  );
         }
         
         StateIterator  end () {
            return StateIterator ( vertices(_net).second , _n2s  );
         }
         
         
         size_t   size() const   {  return  num_vertices ( _net );  }
         
      };


      //
      //    odeint uses boost::size to infer the container's size
      //    boost::size calls a function called range_calculate_size

      template < class Network >
      inline size_t range_calculate_size( const StateProxy< Network > & v )
      {
         return v.size();
      }     


   }     // end detail


   template <
      class   System
     ,class   Coupling
     ,class   SystemState = typename System::state_type
   >
   struct ode_network : public   ode_network_detail::adjacency <
                                    ode_network_detail::system_decorator<System,SystemState> ,
                                    Coupling
                                 > :: list
   {
      typedef
         System
         system_type;

      typedef
         SystemState
         sys_state_type;
      
      typedef
         typename ode_network_detail::adjacency<
                     ode_network_detail::system_decorator<System,SystemState> ,
                     Coupling
                  > :: list
         network;

      typedef
         typename  graph_traits<network>::vertex_descriptor
         Node;
      
      typedef
         typename  graph_traits<network>::vertex_iterator
         NodeIterator;

      typedef
         typename  graph_traits<network>::in_edge_iterator
         InArcIterator;

      typedef
         std::vector < sys_state_type >
         state_type;

      typedef
         ode_network_detail::StateProxy<ode_network>
         StateProxy;




   private:
      
      StateProxy   _state_proxy;
      
      
      
   public:
      
      StateProxy &  state()   {  return  _state_proxy;  }
//      StateProxy    state ()   {  return  StateProxy( *this );  }



      ode_network()
      :  _state_proxy ( *this )
      { }

      ode_network( size_t N_node )
      :  _state_proxy ( *this ),
      network      ( N_node )       // here prob with matrix
      { }


      /* TODO converting constructor that copies the graph topology */
      template < class s , class c >
      ode_network( const typename ode_network_detail::adjacency<s,c>::list & net )
      :  _state_proxy ( *this )
      //: network ( (network)net )
      { }


      template < class s , class c , class t>
      ode_network( const ode_network<s,c,t> & net )
      :  _state_proxy ( *this )
      { }


   private:
/*
      template <
         class  node_descriptor_t,
         class  sub_state_t >

      void call_sub_system(
         const node_descriptor_t    & node ,
         const sub_state_t          & x    ,
         const sub_state_t          & dxdt
      ){

         const network & net = *this
         net[node]( x , dxdt , t );
         
         typename network::in_edge_iterator
         n0 = in_edges ( node , net ). first,
         n1 = in_edges ( node , net ). second;
         while ( n0 != n1 ) {
            //               dxdt[*node]  +=  (*this)[*n0]( x[*node] , x[ source(*n0,*this) ] , t );
            dxdt[node]  +=  net[*n0]( x , _x[ source(*n0,*this) ] , t );
            ++n0;
         }
      }
*/

   public:

/**
      template < class InState , class OutState >
      inline void  operator() ( const InState&  x , OutState &  dxdt , const double  t ) const
      {         
         typename  InState::const_iterator     i = x.begin();
         typename  OutState::iterator          j = dxdt.begin();
         typename  graph_traits<network>::vertex_iterator     node;

         for ( node  = vertices(*this).first;  node != vertices(*this).second;  ++node ) {
            operator[](*node)( *i , *j , t );
            ++i , ++j;
         }
      }
/**/

   private:

      const sys_state_type &  _state_of_node( const StateProxy & x , const Node & n ) const {
         return  (*this)[n].state;     // TODO add []-opertor to StateProxy
      }

      const sys_state_type &  _state_of_node( const state_type & x , const Node & n ) const {
         return  x[n];
      }
      
   public:

      template < class InState , class OutState >
      void  operator() ( const InState&  x , OutState &  dxdt , const double  t ) const
      {
         typename  InState::const_iterator     i = x.begin();
         typename  OutState::iterator          j = dxdt.begin();

         const network & self = *this;

//         for_each ( vertices(*this).first , vertices(*this).second , compute_node );

         NodeIterator  n, n_end;
         boost::tie( n , n_end )  =  vertices(*this);

         while ( n != n_end ) {

            self[*n]( *i , *j , t );

            InArcIterator   a , a_end;
            boost::tie( a , a_end )  =  in_edges ( *n , *this );

            while ( a != a_end ) {
               *j  +=  self[*a]( *i , _state_of_node( x , source(*a,*this) ) , t );
               ++a;
            }
            ++i , ++j, ++n;
         }
      }


   };  // class ode_network


};     // namespace dynamical_system



// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


/*
 *    the code below would be the alternative to the overloaded function
 *    range_calculate_size<StateProxy>() in ode_network_detail above.
 *

namespace odeint {
   
   //
   // some helper struct

   template < class T1 , class T2 >
   struct resize_impl < T1 , dynamical_system::detail::StateProxy<T2> >
   {
      static void resize( T1 & x1 , const dynamical_system::detail::StateProxy<T2> & x2 ) {
         x1.resize( x2.size() );
      }
   };


   template < class T1 , class T2 >
   struct same_size_impl < T1 , dynamical_system::detail::StateProxy<T2> >
   {
      static bool same_size ( const T1  &x1 , const dynamical_system::detail::StateProxy<T2> &x2 ) {
         return ( x1.size() == x2.size() );
      }
   };


   template < class T >
   struct is_resizeable< dynamical_system::detail::StateProxy<T> > {
      typedef  boost::true_type   type;
      const static bool           value = type::value;
   };
   
}
*/



};     // namespace numeric
};     // namespace boost

#endif

