/*    Time series predication (nonlinear map)
 *    • given previous N samples of time series predict next sample
 *    • this model predicts the exact value (floating point)
 *    • be aware of normalization issues with wrongly scaled input
 *
 */

#include <iostream>
#include <fstream>

#include "neural_net.h"
#include "print_tools.h"

#include <vector>
#include <array>
#include <cmath>



int main()
{
   std::vector<float>  signal(20000);

   Span<float>  train_signal{signal.data(),10000}
             ,  test_signal{signal.data()+10000,10000};

   float sig_max = 0.f;
   for (int t=0; t<signal.size(); ++t)
   {
      using std::sin;
      signal[t] = 5. * sin(t/3.7+.3)
                + 3. * sin(t/1.3+.1)
                + 2. * sin(t/34.7+.7)
                + 4. * sin(0.7*t+14*sin(0.1*t))
                + 4. * sin(0.2*t+6*sin(0.02*t));
      sig_max = std::max(sig_max,signal[t]);
   }
   for (auto& x:signal)
      //x /= sig_max;
      x = 0.5*(x/sig_max + 1.);

   constexpr size_t state_size = 40;

   auto training_data = [&]
   {
      using TrainingPair = std::pair<Span<float>,std::array<float,1>>;
      std::vector<TrainingPair>  training_data;

      for (size_t n=state_size; n < train_signal.size(); ++n)
         training_data.push_back({ {train_signal.data()+n-state_size,state_size}, {{train_signal[n]}} });

      return training_data;
   }();


   constexpr size_t n_epochs = 20;

   NeuralNetwork  net({ state_size, 30, 1 });
/*
   auto print_cost = [s = std::ofstream("cost.txt"), &net, &training_data]() mutable
   {
      stat::Averager<float> avg_cost;
      for (auto const& p : training_data) avg_cost(net.cost( net(p.first), p.second ));
      s << avg_cost.get() << std::endl;
   };
*/
   auto prog_bar = print::make_progress_printer(n_epochs,20,"⌛ training network: ");

   auto prog = [&](auto n) mutable
   {
      prog_bar(n);
  //    print_cost();
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


   training_data = [&]
   {
      using TrainingPair = std::pair<Span<float>,std::array<float,1>>;
      std::vector<TrainingPair>  training_data;

      for (size_t n=state_size; n < test_signal.size(); ++n)
         training_data.push_back({ {test_signal.data()+n-state_size,state_size}, {{test_signal[n]}} });

      return training_data;
   }();
   std::cout << "• correct predictions of test set: " << 100.f * prec_predicted(training_data) << "%" << std::endl;

   {
      std::ofstream data_f("data.txt");
      std::vector<float> state(state_size+3);
      for (size_t n=state_size; n<state_size+1000; ++n)
      {
         rng::copy(Span<float>{signal.data()+n-state_size,state_size}, state.begin() );
         state.end()[-3] = net({state.data()+0,state_size})[0];
         state.end()[-2] = net({state.data()+1,state_size})[0];
         state.end()[-1] = net({state.data()+2,state_size})[0];
         data_f << signal[n] << "  " << state.end()[-3] << " " << state.end()[-2] << " " << state.end()[-1] << " " << std::endl;
      }
   }

   {
      std::ofstream data_f("pred.txt");
      auto pred_signal = signal;
      for (size_t n=2*state_size; n<pred_signal.size(); ++n)
         pred_signal[n] = net({ pred_signal.data()+n-state_size,state_size})[0];
      for (size_t n=0; n<signal.size(); ++n)
         data_f << signal[n] << " " << pred_signal[n] << std::endl;
   }




}



