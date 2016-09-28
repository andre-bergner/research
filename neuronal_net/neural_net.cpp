#include <iostream>
#include "neural_net.h"

int main()
{
   /*
   NeuralNetwork  net({4,3,4});

   for ( auto x : net({1,2,3,4}) )
      std::cout << x << std::endl;
   */

   using namespace std;

   //training_data = [ (spice(1)+x,y) for x,y in np.repeat( [ ([ 0.], [1.]), ([ 1.], [0.]) ], 100, axis=0 ) ]
   using vec_t = vector<float>;
   vector<pair<vec_t,vec_t>> training_data;
   mt19937  gen;//(random_device{}());
   //uniform_real_distribution<float>  norm_dist(0.f,0.1f);
   normal_distribution<float>  norm_dist(0.f,0.001f);
   auto rnd = [&]{ return norm_dist(gen); };
   for (size_t n=0; n<100; ++n)
   {
      training_data.push_back({ {0.f + rnd()}, {1.f} });
      training_data.push_back({ {1.f + rnd()}, {0.f} });
   }

   //-------------------------------------------------------------------------

   NeuralNetwork  net({1,1,2,1});

   std::cout << "learning..." << std::endl;
   for (size_t n=0; n<1500; ++n)
      net.step_gradien_descent( training_data, 10.f );
   for (size_t n=0; n<1000; ++n)
      net.step_gradien_descent( training_data, .5f );

   //training_data.clear();
   //training_data.push_back({{0.f},{1.f}});
   //net.step_gradien_descent(training_data, 10.f );

   auto print_back_propagate = [&]{
      auto g = net.back_propagate({1},{0});
      int n = 0;
      for ( auto& l : g )
      {
         std::cout << "layer " << n++ << ":  ";
         for ( auto& x : l.weights )   std::cout << x << ", ";
         for ( auto& x : l.biases )    std::cout << x << ", ";
         std::cout << std::endl;
      }
   };

   std::cout << "-----" << std::endl;
   std::cout << net({0.f})[0] << std::endl;
   std::cout << net({1.f})[0] << std::endl;
   print_back_propagate();
/*
   training_data.clear();
   training_data.push_back({{1.f},{0.f}});
   net.step_gradien_descent(training_data, 1.f );

   std::cout << "-----" << std::endl;
   std::cout << net({0.f})[0] << std::endl;
   std::cout << net({1.f})[0] << std::endl;
   print_back_propagate();
*/
}

