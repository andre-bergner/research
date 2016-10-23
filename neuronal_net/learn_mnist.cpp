#include <iostream>
#include "neural_net.h"

#include <jeayeson/jeayeson.hpp>

#include <vector>
#include <array>
#include <iostream>
#include <iomanip>



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
         rng::transform(img,p.first,[](auto& x){ return float(x.template as<json_int>())/255.f; });
         training_data.push_back(std::move(p));
      }

      return training_data;
   };



   std::cout << "loading data file." << std::endl;

   auto training_data = load_mnist("data0.json");

   const size_t img_size = training_data[0].first.size();



   auto make_progress_printer = []( size_t n_max )
   {
      return [n_max](size_t n)
      {
         auto perc = float(n+1) / float(n_max);
         std::cout << "\r";
         std::cout << "[" << std::string(int(20*perc),'*')
                          << std::string(20-int(20*perc),' ')
                   << "]  ";
         std::cout << n+1 << "/" << n_max << " (" << int(100.f*perc) << "%)";
         std::cout << std::flush;
      };
   };



   size_t n_epochs = 30;

   NeuralNetwork  net({ img_size, 30, 10 });

   auto print_cost = [s = std::ofstream("cost.txt"), &net, &training_data]() mutable
   {
      stat::Averager<float> avg_cost;
      for (auto const& p : training_data) avg_cost(net.cost( net(p.first), p.second ));
      s << avg_cost.get() << std::endl;
   };

   auto prog = [&,f=make_progress_printer(n_epochs)](auto n)mutable{f(n);print_cost();};

   std::cout << "training network" << std::endl;
   stochastic_gradien_descent(net, training_data, {n_epochs, 10, 3., prog});


   std::cout << "\n testing..." << std::endl;

   auto max_idx = [](auto const& xs){
      auto me = std::max_element(xs.begin(),xs.end());
      return *me > 0.5f ? int(std::distance(xs.begin(),me)) : -1;
   };


   std::cout << "same set -----------" << std::endl;

   for (size_t n=0; n < 20; ++n )
      std::cout << max_idx(training_data[n].second) << "   vs    " << max_idx(net( training_data[n].first )) << std::endl;

   std::cout << "other set -----------" << std::endl;
   training_data = load_mnist("data2.json");

   for (size_t n=0; n < 10; ++n )
      std::cout << max_idx(training_data[n].second) << "   vs    " << max_idx(net( training_data[n].first )) << std::endl;
}




