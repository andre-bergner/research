#include "matrix.h"
#include "algorithm.h"

#include <vector>
#include <array>
#include <memory>
#include <cmath>

// Types of Layers:
//
//    1) fixed number of inputs and outputs (classical all-to-all connection, fixed matrix)
//    2) fixed input/output ratio:
//       a) #input = #output
//          I)  pure functional mapping (e.g. sigmdoial, relu,...)
//          II) symmetric all to all couplings (e.g. Softmax)
//       b) #input = ratio * #output
//          Convolutional Layer


// TODO: copy Span from xlibs -- should support initilizer_list etc..


namespace baklava
{

// TODO: get UNIVERSAL FUNCTION from xlibs

struct error_t {};
struct free_function_t   : error_t {};
struct member_function_t : free_function_t {};


namespace detail {

   template <typename L>
   auto in_size( free_function_t, L const& l ) -> decltype( l.in_size() ) { return l.in_size(); }

   template <typename L>
   auto in_size( member_function_t, L const& l ) -> decltype( in_size(l) ) { return in_size(l); }

   template <typename L>
   auto out_size( free_function_t, L const& l ) -> decltype( l.out_size() ) { return l.out_size(); }

   template <typename L>
   auto out_size( member_function_t, L const& l ) -> decltype( out_size(l) ) { return out_size(l); }

   template <typename L, typename T>
   auto apply( free_function_t, L const& l, Span<T const> in, Span<T> out) -> decltype( apply(l,in,out) ) { apply(l,in,out); }

   template <typename L, typename T>
   auto apply( member_function_t, L const& l, Span<T const> in, Span<T> out) -> decltype(l.apply(in,out)) { l.apply(in,out); }


   template <typename L>
   auto has_nonlinear_derivative( free_function_t, L const& l) -> decltype( has_nonlinear_derivative(l) ) { return has_nonlinear_derivative(l); }

   template <typename L>
   auto has_nonlinear_derivative( member_function_t, L const& l) -> decltype(l.has_nonlinear_derivative()) { return l.has_nonlinear_derivative(); }

   template <typename L, typename T>
   auto multiply_backward_derivative( free_function_t, L const& l, Span<T> inout) -> decltype( multiply_backward_derivative(l,inout) ) { multiply_backward_derivative(l,inout); }

   template <typename L, typename T>
   auto multiply_backward_derivative( member_function_t, L const& l, Span<T> inout) -> decltype(l.multiply_backward_derivative(inout)) { l.multiply_backward_derivative(inout); }

}

template <typename L>
auto adl_in_size( L const& l ) { return detail::in_size( member_function_t{}, l ); }

template <typename L>
auto adl_out_size( L const& l ) { return detail::out_size( member_function_t{}, l ); }

template <typename L, typename T>
auto adl_apply( L const& l, Span<T const> in, Span<T> out) { return detail::apply(member_function_t{}, l,in, out); }


template <typename L>
auto adl_has_nonlinear_derivative( L const& l) { return detail::has_nonlinear_derivative(member_function_t{}, l); }

template <typename L, typename T>
auto adl_multiply_backward_derivative( L const& l, Span<T> inout) { return detail::multiply_backward_derivative(member_function_t{}, l,inout); }



template <typename T>
struct LayerConcept
{
   virtual ~LayerConcept() = default;
   virtual LayerConcept* clone() const = 0;

   virtual size_t in_size_() const = 0;
   virtual size_t out_size_() const = 0;
   virtual void   apply_( Span<T const>, Span<T> ) const = 0;

   virtual bool   has_nonlinear_derivative_() const = 0;
   virtual void   multiply_backward_derivative_( Span<T> ) const = 0;
};


template <typename T, typename Model>
struct LayerModel final : LayerConcept<T>
{
   Model m;

   LayerModel( Model m_ ) : m(std::move(m_)) {}

   LayerModel* clone() const override { return new LayerModel(*this); }

   size_t in_size_() const override   { return adl_in_size(m); }
   size_t out_size_() const override  { return adl_out_size(m); }

   void   apply_( Span<T const> in, Span<T> out ) const override
   { adl_apply(m, in, out); }


   bool has_nonlinear_derivative_() const override
   { return adl_has_nonlinear_derivative(m); }

   void   multiply_backward_derivative_( Span<T> inout ) const override
   { adl_multiply_backward_derivative(m, inout); }
};


template <typename T>
class Layer
{
   std::unique_ptr<LayerConcept<T>> l_;

public:

   template <typename Model>
   Layer( Model m ) : l_{ std::make_unique<LayerModel<T,Model>>(std::move(m)) } {}

   Layer( Layer const & l ) : l_{ l.l_->clone() } {}

   size_t map_size(size_t in_size) const;    // TODO use this instead of (in/out)_size ???
   size_t in_size() const  { return l_->in_size_(); }
   size_t out_size() const { return l_->out_size_(); }

   void   apply( Span<T const> in, Span<T> out ) const
   { return l_->apply_(in, out); }


   bool has_nonlinear_derivative() const
   { return l_->has_nonlinear_derivative_(); }

   void   multiply_backward_derivative( Span<T> inout ) const
   { return l_->multiply_backward_derivative_(inout); }
};




// default implementation

template <typename Layer>
bool has_nonlinear_derivative(Layer const& l) { return true; }






//   -----------------------------------------------------------------------------------------------
//  Layer Algorithms
//   -----------------------------------------------------------------------------------------------

template <typename LayerRange, typename Value >
auto feed( LayerRange const& layers, Span<const Value> input )
{
   std::vector<Value> output, temp;

   for ( auto const& l : layers )
   {
      auto size = l.out_size() > 0 ? l.out_size() : input.size();
      output.resize(size);     // TODO should be uninitialized and not copy!!!
      l.apply( input, output );
      input = output;
      std::swap(output,temp);
   }

   return temp;
}



template <typename LayerRange, typename Value >
auto back_propagate( LayerRange const& layers, Span<const Value> input, Span<const Value> output )
{
   // 1. propagate forward and store layer results.

   using result_t = std::vector<Value>;

   std::vector<result_t> layer_output{ result_t(input.size()) };
   rng::copy(input, layer_output.back().begin());

   for (auto const& l : layers)
   {
      auto size = l.out_size() > 0 ? l.out_size() : layer_output.back().size();
      result_t l_output(size);
      l.apply( layer_output.back(), l_output );
      layer_output.push_back( std::move(l_output) );
   }

   // 2. propagate backword and compute derivatives.



}















//   -----------------------------------------------------------------------------------------------
//  Linear Mixing Layer
//   -----------------------------------------------------------------------------------------------



template <typename T>
class LinearMixing
{
   using value_t = T;
   using vec_t   = std::vector<value_t>;
   using mat_t   = matrix<value_t>;

public:

   // TODO public mutable span on biases and weights (changable but not resizable)

   mat_t   weights;
   vec_t   biases;

public:

   LinearMixing( size_t in_size , size_t out_size )
   :  weights(out_size,in_size)
   ,  biases(out_size)
   {}

   size_t in_size() const  { return weights.col_size(); }      // actuall not needed !?
   size_t out_size() const { return weights.row_size(); }      // instead info(): returns info object (key,value)

   LinearMixing& operator=( std::initializer_list<T> const& values )
   {
      weights = values;
      return *this;
   }

   void apply( Span<T const> in, Span<T> out) const
   {
      assert( in.size() == in_size() );
      assert( out.size() == out_size() );

      rng::copy( biases, out.begin() );
      dot_add( weights, in, out );
   }

   bool has_nonlinear_derivative() const { return false; }

   void multiply_backward_derivative( Span<T> ) const
   {

   }
};












//   -------------------------------------------------------------------------------------
//  Sigmoidal Layer
//   -------------------------------------------------------------------------------------

template <typename T>
auto func(T const& x) { return T{1} / (T{1} + std::exp(-x)); };

template <typename T>
auto dfunc(T const& x) { return func(x) * (T{1}-func(x)); };


class Sigmoidal
{
public:

   size_t in_size() const  { return 0; }  // means any, for now
   size_t out_size() const { return 0; }  // means any, for now

   // TODO generalize interface to remove unnecassary copy

   template <typename T>
   void apply( Span<T const> in, Span<T> out) const
   {
      auto it = out.begin();
      for ( auto& x : in )
         *it++ = func(x);
   }

   template <typename T>
   void multiply_backward_derivative( Span<T> xs ) const
   {
      for ( auto& x : xs ) x *= dfunc(x);      
   }
};

















template <typename T>
class Function
{
   using value_t = T;

public:

   size_t in_size() const  { return 0; }  // means any, for now
   size_t out_size() const { return 0; }  // means any, for now

   void apply( Span<T const> in, Span<T> out) const
   {
      //for ( auto x )
   }

   void multiply_backward_derivative( Span<T> ) const
   {

   }
};


struct Sin {  template <typename T> T operator()( T&& x ) { std::sin(x); } };
struct Cos {  template <typename T> T operator()( T&& x ) { std::cos(x); } };


template <typename Function>
struct Derivative;

template <> struct Derivative<Sin> : Cos {};



}