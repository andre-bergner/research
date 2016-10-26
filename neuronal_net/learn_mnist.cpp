#include <iostream>

#include "neural_net.h"
#include "print_tools.h"

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


   auto training_data = [&]
   {
      print::RunningWheel wheel;
      std::cout << "  loading data file." << std::flush;
      return load_mnist("data0.json");
   }();

   const size_t img_size = training_data[0].first.size();
   std::cout << std::endl;



   size_t n_epochs = 30;

   NeuralNetwork  net({ img_size, 30, 20, 10 });

   auto print_cost = [s = std::ofstream("cost.txt"), &net, &training_data]() mutable
   {
      stat::Averager<float> avg_cost;
      for (auto const& p : training_data) avg_cost(net.cost( net(p.first), p.second ));
      s << avg_cost.get() << std::endl;
   };

   auto prog_bar = print::make_progress_printer(n_epochs,20,"⌛ training network: ");

   auto prog = [&](auto n) mutable
   {
      prog_bar(n);
      print_cost();
   };

   {
      print::RunningWheel wheel;
      prog_bar(0);
      stochastic_gradien_descent(net, training_data, {n_epochs, 10, 3., prog});
   }
   std::cout << std::endl;
   
   std::cout << "• testing..." << std::endl;

   auto max_idx = [](auto const& xs){
      auto me = std::max_element(xs.begin(),xs.end());
      return *me > 0.5f ? int(std::distance(xs.begin(),me)) : -1;
   };

   auto prec_predicted = [&max_idx,&net]( auto const& pairs )
   {
      auto correct_prediction = [&](auto const& t){ return max_idx(t.second) == max_idx(net(t.first)); };
      return float(rng::count_if(pairs, correct_prediction)) / float(pairs.size());
   };

   std::cout << "• correct predictions within learned set: " << 100.f * prec_predicted(training_data) << "%" << std::endl;

   training_data = load_mnist("data2.json");
   std::cout << "• correct predictions of test set: " << 100.f * prec_predicted(training_data) << "%" << std::endl;


}







