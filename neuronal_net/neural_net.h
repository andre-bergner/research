#include "matrix.h"
#include "algorithm.h"

#include <boost/iterator/permutation_iterator.hpp>
#include <boost/range/iterator_range.hpp>

#include <initializer_list>
#include <random>
#include <cmath>
#include <functional>


// TODO
// • custom allocator used by matrices & vectors → all memory is in one piece
// • visitor gets passed into back_propagate function to update weights in place
// • use span<> for inputs


class NeuralNetwork
{
   using value_t = float;

   using vec_t  = std::vector<value_t>;
   using cvec_t = vec_t const&;
   using rvec_t = vec_t&;

   using mat_t = matrix<value_t>;
   using span_t = Span<value_t>;
   using cspan_t = Span<value_t const>;

   using init_t = std::initializer_list<value_t>;

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

      std::function<void(span_t)>  activation;
      std::function<void(cspan_t,span_t)>  multiply_nabla_activation;
   };

   std::vector<Layer>  layers_;        // name: baklava

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
   void feed_layer( Layer const& l, cspan_t in, span_t out, F&& visit_node_input ) const
   {
      rng::copy( l.biases, out.begin() );
      dot_add( l.weights, in, out );
      visit_node_input(out);
      l.activation(out);
   }

   void feed_layer( Layer const& l, cspan_t in, span_t out ) const
   {
      feed_layer(l,in,out,[](auto&){});
   }


public:

   NeuralNetwork( std::initializer_list<std::size_t> layer_sizes)
   :  temp1_ ( *std::max_element(layer_sizes.begin()+1, layer_sizes.end()) )
   ,  temp2_ ( temp1_.size() )
   {
      using namespace std;
      mt19937  gen(random_device{}());
      normal_distribution<value_t>  norm_dist;
      auto rnd = [&]{ return norm_dist(gen); };

      // transform( layer_sizes | windowed(2,1), back_inserter(layers_), construct<Layer>{} )

      auto s = layer_sizes.begin();
      for (; s != std::prev(layer_sizes.end()) ;)
      {
         auto in_size = *s++;
         Layer l(in_size,*s);
         rng::generate(l.weights, rnd);
         rng::generate(l.biases, rnd);
         l.activation = [this](span_t xs){ for (auto& x : xs) x = func(x); };
         l.multiply_nabla_activation = [this](cspan_t xs, span_t ys){
            for (size_t n=0; n<ys.size(); ++n)
               ys[n] *= dfunc(xs[n]);
         };
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

   cspan_t operator()( cspan_t input ) const
   {
      cspan_t in   = input;
      span_t  out  = temp1_;
      span_t  temp = temp2_;

      // activation(wu+b)
      for ( auto const& l : layers_ )
      {
         feed_layer(l, in, out);
         in = out;
         std::swap(out,temp);
      }
      return {in.data(),layers_.back().out_size() };
   }

   cspan_t operator()( init_t input ) const
   {
      vec_t v(input);
      return (*this)(v);
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


   //template <typename Visitor>
   decltype(auto) back_propagate( cspan_t x, cspan_t expected )//, Visitor&& update_coeff )
   {
      // 1. feed forward and store all node input and outputs

      struct web_in_out {   // better name: axons? axon_web? tissue!
         vec_t  in;         // output of prev layer, i.e. post neuron
         vec_t  out;        // input to next layer, i.e. pre neuron
      };

      std::vector<web_in_out>  webs;
      webs.reserve(layers_.size());

      vec_t in(x.size());
      rng::copy(x,in.begin());

      for ( auto const& l : layers_ )
      {
         web_in_out w{ std::move(in), .out = vec_t(l.out_size()) };
         in = vec_t(l.out_size());
         feed_layer( l, w.in, in, [&](auto& v){ rng::copy(v,w.out.begin()); });
         webs.push_back(std::move(w));
      }


      // 2. feed backward -- back propagation

      // TODO introduce pop_back_iterator for web

      auto& dlayers = dlayers_;
      auto it_dl = dlayers.rbegin();
      auto& error = it_dl->biases;
      for ( int n=0; n < in.size(); ++n )
         error[n] = nabla_cost(in[n], expected[n]);
      it_dl->multiply_nabla_activation(webs.back().out,error);
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

         it_dl->multiply_nabla_activation(webs.back().out,error);

         outer(webs.back().in,error,it_dl->weights);
      }

      return dlayers;
   }

   decltype(auto) back_propagate( init_t x, init_t expected )
   {
      vec_t x_{x};
      vec_t e_{expected};
      return back_propagate(x_,e_);
   }


   template <typename TrainingPairs>
   void step_gradient_descent( TrainingPairs const& pairs, value_t eta = 0.1f )
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
      auto norm = eta / value_t(pairs.size());
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



struct sgd_params
{
   size_t n_epochs        = 100;
   size_t mini_batch_size = 20;
   float  eta             = 0.1f;
   std::function<void(size_t)>  progress = [](size_t){};
};

template <typename TrainingPairs>
void stochastic_gradien_descent( NeuralNetwork& net, TrainingPairs const& pairs, sgd_params p = {} )
{
   using namespace std;
   mt19937  gen(random_device{}());
   uniform_int_distribution<size_t>  uni_dist( 0, pairs.size() );
   auto rnd_idx = [&]{ return uni_dist(gen); };

   for (size_t n=0; n<p.n_epochs; ++n)
   {
      auto n_batches = pairs.size() / p.mini_batch_size;
      for (size_t k=0; k<n_batches; ++k)
      {
         std::vector<size_t> indices(p.mini_batch_size);
         rng::iota(indices,0);
         rng::shuffle(indices,gen);

         auto it = boost::make_permutation_iterator(pairs.begin(), indices.begin());
         auto jt = boost::make_permutation_iterator(pairs.begin(), indices.end());

         net.step_gradient_descent( boost::make_iterator_range(it,jt), p.eta );
      }
      p.progress(n);
   }
}

