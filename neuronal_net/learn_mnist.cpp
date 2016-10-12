#include <iostream>
#include "neural_net.h"

#include <jeayeson/jeayeson.hpp>

#include <vector>
#include <array>
#include <iostream>
#include <iomanip>

template <typename ContiguousRange>
auto as_span( ContiguousRange const& r )
{
   using value_t = typename ContiguousRange::value_type;
   return Span<const value_t>( &r[0], r.size() );
}

template <typename Range>
std::ostream& print_range( std::ostream& os, Range const& xs )
{
   for ( auto const& x : xs ) os << x << ", ";
   return os;
}

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

template <typename T, size_t N>
std::ostream& operator<<( std::ostream& os, std::array<T,N> const& vec )
{
   for ( auto const& x : vec ) os << x << ", ";
   return os;
}


int main()
{

   auto load_mnist = [](std::string file_name = "data2.json")
   {
      using namespace jeayeson;

      json_map  mnist2{ json_file{ file_name } };

      auto& data = mnist2["data"].as<json_array>();

      using TrainingPair = std::pair<std::vector<float>,std::array<float,10>>;
      std::vector<TrainingPair>  training_data;
      training_data.reserve( data.size() );

      for ( auto& pair : data )
      {
         auto& img = pair["image"].as<json_array>();
         int num = pair["number"].as<json_int>();
         TrainingPair p;
         p.first.resize(img.size());
         p.second.fill(0);
         p.second[num] = 1;
         rng::transform(img,p.first,[](auto& x){return x.template as<json_int>();});
         training_data.push_back(std::move(p));
      }

      return training_data;
   };



   std::cout << "loading data file." << std::endl;

   auto training_data = load_mnist("data0.json");

   const size_t img_size = training_data[0].first.size();



   size_t n_epochs = 10;

   auto progress_printer = [n_max = n_epochs](size_t n)
   {
      auto perc = float(n) / float(n_max);
      std::cout << "\r";
      std::cout << "[" << std::string(int(20*perc),'*')
                       << std::string(20-int(20*perc),' ')
                << "]  ";
      std::cout << n << "/" << n_max << " (" << int(100.f*perc) << "%)";
      std::cout << std::flush;
   };



   NeuralNetwork  net({ img_size, 30, 10 });

   std::cout << "training network" << std::endl;
   stochastic_gradien_descent(net, training_data, {n_epochs, 10, 3., progress_printer});
   std::cout << std::endl;


   std::cout << "\n testing..." << std::endl;

   auto max_idx = [](auto const& xs){
      auto me = std::max_element(xs.begin(),xs.end());
      return *me > 0.5f ? int(std::distance(xs.begin(),me)) : -1;
   };


   std::cout << "same set -----------" << std::endl;

   for (size_t n=0; n < 10; ++n )
      std::cout << max_idx(training_data[n].second) << "   vs    " << max_idx(net( training_data[n].first )) << std::endl;

   std::cout << "other set -----------" << std::endl;
   training_data = load_mnist("data1.json");

   for (size_t n=0; n < 10; ++n )
      std::cout << max_idx(training_data[n].second) << "   vs    " << max_idx(net( training_data[n].first )) << std::endl;
}




