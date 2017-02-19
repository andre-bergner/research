#include "baklava.hpp"
#include "algorithm.h"

#include <iostream>
#include <vector>
#include <random>

namespace baklava
{
   template <typename Value = float>
   auto make_random_mixing_layer( size_t in_size, size_t out_size )-> LinearMixing<Value>
   {
      using namespace std;
      mt19937  gen(random_device{}());
      normal_distribution<Value>  norm_dist;
      auto rnd = [&]{ return norm_dist(gen); };

      LinearMixing<Value> l(in_size,out_size);
      rng::generate(l.weights, rnd);
      rng::generate(l.biases, rnd);
      return l;
   }
}


int main()
{
   using namespace baklava;
   using namespace std;

   // TODO: train using old network --> check output

   vector<Layer<float>>  layers =
   {  make_random_mixing_layer(4,3)
   ,  Sigmoidal{}
   ,  make_random_mixing_layer(3,2)
   ,  Sigmoidal{}
   };

   vector<float> xs = { 0.1f, 0.5f, 100.f, -20.f };

   auto ys = feed(layers, Span<const float>(xs));

   for ( auto x : ys) cout << x << ", ";
   cout << endl;


   auto gradient = back_propagate(layers, Span<const float>(xs), Span<const float>(ys));


   // Back propagation
   // 1. feed fwd & store restult !
   // 2. compute error in final layer as starting point
   // 3. feed bwd using error as initial vector and stored results
}