#include "matrix.h"
#include "algorithm.h"

#include <initializer_list>
#include <random>
#include <cmath>


// TODO
// • custom allocator used by matrices & vectors → all memory is in one piece
// • visitor gets passed into back_propagate function to update weights in place
// • use span<> for inputs


class NeuralNetwork
{
   using vec_t  = std::vector<float>;
   using cvec_t = vec_t const&;
   using rvec_t = vec_t&;

   using mat_t = matrix<float>;
   using span_t = Span<float>;
   using cspan_t = Span<float const>;

   struct Layer {

      Layer() = default;

      Layer(size_t in_size, size_t out_size)
      :  weights(out_size,in_size)
      ,  biases(out_size)
      {}

      mat_t   weights;
      vec_t   biases;

      size_t in_size() const  { return weights.col_size(); }
      size_t out_size() const { return weights.row_size(); }
   };

   std::vector<Layer>  layers_;

   mutable vec_t  temp1_, temp2_;

   std::vector<Layer>  dlayers_;
   std::vector<Layer>  deltas_sum_;


   // activation function and derivative

   template <typename T>
   auto func(T const& x) const { return T{1} / (T{1} + std::exp(-x)); };

   template <typename T>
   auto dfunc(T const& x) const { return func(x) * (T{1}-func(x)); };



   template <typename T>
   auto nabla_cost(T const& v, T const& y) const  { return  v - y; };


   template <typename F>
   void feed_layer( Layer const& l, cvec_t in, rvec_t out, F&& visit_node_input ) const
   {
      rng::copy( l.biases, out.begin() );
      dot_add( l.weights, in, out );
      visit_node_input(out);
      for ( auto& x : out ) x = func(x);
   }

   void feed_layer( Layer const& l, cvec_t in, rvec_t out ) const
   {
      feed_layer(l,in,out,[](auto&){});
   }


public:

   NeuralNetwork( std::initializer_list<std::size_t> layer_sizes)
   :  temp1_ ( *std::max_element(layer_sizes.begin()+1, layer_sizes.end()) )
   ,  temp2_ ( temp1_.size() )
   {
      using namespace std;
      mt19937  gen;//(random_device{}());
      normal_distribution<float>  norm_dist;
      auto rnd = [&]{ return norm_dist(gen); };

      // transform( layer_sizes | windowed(2,1), back_inserter(layers_), construct<Layer>{} )

      auto s = layer_sizes.begin();
      for (; s != std::prev(layer_sizes.end()) ;)
      {
         auto in_size = *s++;
         Layer l(in_size,*s);
         rng::generate(l.weights, rnd);
         rng::generate(l.biases, rnd);
         layers_.push_back(std::move(l));
      }

      // temp vec for one learning step
      dlayers_ = layers_;
      deltas_sum_ = layers_;
      for (auto& l : deltas_sum_) {
         rng::fill( l.weights, 0.f );
         rng::fill( l.biases, 0.f );
      }
   }

   cspan_t operator()( cvec_t input ) const
   {
      auto* in   = &input;
      auto* out  = &temp1_;
      auto* temp = &temp2_;

      // activation(wu+b)
      for ( auto const& l : layers_ )
      {
         feed_layer(l, *in, *out);
         in = out;
         std::swap(out,temp);
      }
      return {in->data(),layers_.back().out_size() };
   }


   template <typename Range1, typename Range2>
   auto cost(Range1 const& v, Range2 const& y) const
   {
      auto sqr = [](auto const& x){ return x*x; };
      using value_t = std::decay_t<decltype(v[0])>;
      value_t sum = {};
      for ( size_t n=0 ; n < v.size() ; ++n )
         sum += sqr(v[n]-y[n]);
      return sum / value_t(2*v.size());
   };


   //template <typename F>
   decltype(auto) back_propagate( cvec_t x, cvec_t y)//, F&& update_coeff )
   {
      // 1. feed forward and store all node input and outputs

      struct web_in_out {
         vec_t  in;         // output of prev layer, i.e. post neuron
         vec_t  out;        // input to next layer, i.e. pre neuron
      };

      std::vector<web_in_out>  webs;
      webs.reserve(layers_.size());

      vec_t in = x;

      for ( auto const& l : layers_ )
      {
         web_in_out w{ std::move(in), .out = vec_t(l.out_size()) };
         in = vec_t(l.out_size());
         feed_layer( l, w.in, in, [&](auto& v){ w.out = v; });
         webs.push_back(std::move(w));
      }


      // 2. feed backward -- back propagation

      auto& dlayers = dlayers_;
      auto it_dl = dlayers.rbegin();
      auto& error = it_dl->biases;
      for ( int n=0; n < in.size(); ++n )
         error[n] = nabla_cost(in[n],y[n]) * dfunc(webs.back().out[n]);
      outer(webs.back().in,error,it_dl->weights);
      ++it_dl;

      webs.pop_back();

      auto* error_pre = &error;
      auto it_w = layers_.rbegin();
      for (; !webs.empty(); ++it_w, ++it_dl, webs.pop_back() )
      {
         auto& error = it_dl->biases;
         dott(it_w->weights, *error_pre, error);
         error_pre = &error;

         for (size_t n=0; n<error.size(); ++n)
            error[n] *= dfunc(webs.back().out[n]);

         outer(webs.back().in,error,it_dl->weights);
      }

      return dlayers;
   }


   void step_gradien_descent( std::vector<std::pair<vec_t,vec_t>> const& pairs, float eta = 0.1f )
   {
      using namespace std;

      auto& deltas_sum = deltas_sum_;
      for (auto& l : deltas_sum) {
         rng::fill( l.weights, 0.f );
         rng::fill( l.biases, 0.f );
      }

      auto add_vec = [](auto& v1, auto const& v2)
      {
         transform( begin(v1), end(v1), begin(v2), begin(v1), std::plus<>{} );
      };

      // compute gradient for all training pairs
      for (auto const& p : pairs)
      {
         auto deltas = back_propagate(p.first, p.second);
         auto it_ds = deltas_sum.begin();
         for (auto const& d : deltas)     // TODO use zip
         {
            add_vec( it_ds->weights, d.weights );
            add_vec( it_ds->biases, d.biases );
            ++it_ds;
         }
      }

      // apply average gradient on coefficients
      auto it_l = layers_.begin();
      auto norm = eta / float(pairs.size());
      for (auto const& d : deltas_sum)     // TODO use zip:  for(auto z:zip(layers_,deltas_sum)) z[_1] -= 
      {
         transform( begin(it_l->weights), end(it_l->weights), begin(d.weights), begin(it_l->weights)
                  , [norm](auto const& x, auto const& dx){ return x - dx * norm; });
         transform( begin(it_l->biases), end(it_l->biases), begin(d.biases), begin(it_l->biases)
                  , [norm](auto const& x, auto const& dx){ return x - dx * norm; });
         ++it_l;
      }
   }

};
