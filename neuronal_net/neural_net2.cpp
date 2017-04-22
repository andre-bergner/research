#include <iostream>

#include "neural_net.h"
#include <fstream>

template <typename T>
std::ostream& operator<<( std::ostream& os, std::vector<T> const& vec )
{
   for ( auto const& x : vec ) os << x << ", ";
   return os;
}

template <typename T>
std::ostream& operator<<( std::ostream& os, Span<T> const& vec )
{
   for ( auto const& x : vec ) os << x << ", ";
   return os;
}


int main()
{
   using namespace std;

   using vec_t = vector<float>;
   vector<pair<vec_t,vec_t>> training_data;
   mt19937  gen(random_device{}());
   normal_distribution<float>  norm_dist(0.f,0.01f);
   auto rnd = [&]{ return norm_dist(gen); };

   for (size_t n=0; n<100; ++n)
   {
      training_data.push_back({ {1.f + rnd(), 1.f + rnd(), 0.f + rnd(), 0.f + rnd()}, {1.f,0.f} });
      training_data.push_back({ {1.f + rnd(), 0.f + rnd(), 1.f + rnd(), 0.f + rnd()}, {0.f,1.f} });
      training_data.push_back({ {1.f + rnd(), 0.f + rnd(), 1.f + rnd(), 1.f + rnd()}, {.0f,.0f} });
   }

   //-------------------------------------------------------------------------

   NeuralNetwork  net({4,3,2});


   std::cout << "learning..." << std::endl;

   auto print_cost = [s = std::ofstream("cost.txt"), &net, &training_data]() mutable
   {
      stat::Averager<float> avg_cost;
      for (auto const& p : training_data) avg_cost(net.cost( net(p.first), p.second ));
      s << avg_cost.get() << std::endl;
   };

   stochastic_gradien_descent(net, training_data, {200,10,0.5,[&](auto){print_cost();}});

   std::cout << "--- some predictions ---" << std::endl;

   std::cout << net({1.f, 1.f, 0.f, 0.f}) << std::endl;
   std::cout << net({1.f, 0.f, 1.f, 0.f}) << std::endl;
   std::cout << net({1.f, 0.f, 1.f, 1.f}) << std::endl;
   std::cout << net({0.f, 1.f, 1.f, 0.f}) << std::endl;

}

