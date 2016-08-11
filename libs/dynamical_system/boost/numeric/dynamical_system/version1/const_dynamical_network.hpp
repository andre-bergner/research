#ifndef		_DYNAMICAL_SYSTEM_NETWORK_
#define		_DYNAMICAL_SYSTEM_NETWORK_


#include	"../const_dynamical_system.hpp"
#include	"const_network.hpp"


namespace boost {
namespace numeric {
namespace dynamical_system {


   template <
      class   _system,
      size_t  _N_node
   >
   class const_dynamical_network :
      public  const_dynamical_system< _system::dimension*_N_node , typename _system::state_type >,
      public  const_network< _system , _N_node >
   {
   public:
      using  const_network< _system , _N_node >::in_node;
      using  const_network< _system , _N_node >::node;

      typedef
         typename  _system::state
            sub_state;

/*
      typedef
         typename  const_dynamical_system< _system::dimension*_N_node , typename _system::state_type >::state
            state_base;

      struct state : public state_base {
         typename  _system::state_type & operator() ( const size_t n , const size_t d )
         {  return this->state_base::operator[]( n * _system::dimension + d );  }
//         {  return (*this)[ n * _system::dimension + d ];  }

      	// FIXME this is a dirty hack, has to be generalized somehow
//         sub_state& operator[] ( const size_t n )
//         {  return  *( (sub_state*) ( &(*this)[n*_system::dimension] ) );  }

//         sub_state& operator[] ( const size_t n ) const
//         {  return  *( (sub_state*) ( &(*this)[n*_system::dimension] ) );  }

      };
*/


      typedef
         boost::array< typename _system::state , _N_node >
            state;


/*
      // FIXME iterator over single elements is needed
      typedef
         std::tr1::array< typename _system::state , _N_node >
            state_base;

      struct state : public state_base {
         typename _system::state_type  begin() { return  this->state_base::begin().begin(); }
         typename _system::state_type  end() { return  this->state_base::end().end(); }

      };
*/


      void  operator()  (
         const state &  x ,
               state &  y ,
         const double   t )
      {
         for ( size_t n = 0  ;  n < _N_node  ;  ++n )
            node(n)(
//               *((sub_state*)(&x[n*_system::dimension])),	// FIXME this is a dirty hack, has to be generalized somehow
//               *((sub_state*)(&y[n*_system::dimension])),	// FIXME same here
               x[n],
               y[n],
               t,
               _system :: coupling_function ( x , n , in_node(n).begin() , in_node(n).end() , glob_coupl )
            );
      }
/*
      void  process_state ( state & x )
      {
         for ( int n=0; n<_N_node; ++n ) 
            node(n).process_state( *((sub_state*)(&x[n*_system::dimension])) );
      }
*/

   protected:
      double   glob_coupl;

   public:
      void setGlobalCoupling   ( double val )   { glob_coupl = val; }
      void set_global_coupling ( double val )   { glob_coupl = val; }
      void setCoupling         ( double val )   { glob_coupl = val; }
      void set_coupling        ( double val )   { glob_coupl = val; }


   };  // class const_dynamical_network

};     // namespace dynamical_system
};     // namespace boost
};     // namespace numeric


#endif

