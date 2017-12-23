/*    Time series predication (nonlinear map)
 *    • given previous N samples of time series predict next sample
 *    • this model predicts the probability distribution of the output sample (histogram, bins)
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

   constexpr size_t state_size = 50;
   constexpr size_t num_bins = 32;


   using bins_t = std::array<float,num_bins>;
   using TrainingPair = std::pair<Span<float>, bins_t>;
   //auto prob = [](float x){ bins_t bs = {{}}; bs[size_t((num_bins-1)*x)] = 1.f; return bs; };
   auto prob = [](float x)
   {
      bins_t bs = {{}};
      float b = (num_bins-1) * x;
      int n = size_t(b);
      float m = b - float(n);
      bs[n]   = 1.f - m;
      bs[n+1] = m;
      return bs;
   };

   auto training_data = [&]
   {
      std::vector<TrainingPair>  training_data;

      for (size_t n=state_size; n < train_signal.size(); ++n)
         training_data.push_back({ {train_signal.data()+n-state_size,state_size}, prob(train_signal[n]) });

      return training_data;
   }();
/*
   for (size_t n=0; n<10; ++n)
   {
      auto& p = training_data[n];
      std::cout << "---------------------" << std::endl;
      for (auto x : p.first) std::cout << x << ", "; std::cout << std::endl;
      for (auto x : p.second) std::cout << x << ", "; std::cout << std::endl;
   }
*/
   constexpr size_t n_epochs = 20;

   NeuralNetwork  net({ state_size, 30, num_bins });
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
      //return *me > 0.5f ? int(std::distance(xs.begin(),me)) : -1;
      return int(std::distance(xs.begin(),me));
   };

   auto mean_idx = [](auto const& xs){
      std::vector<float> xs_copy(xs.begin(),xs.end());
      auto sum = rng::accumulate(xs,0.f);
      float f = 0.f;
      for ( auto& x:xs_copy) { x *= f; ++f; }
      return int(std::round(rng::accumulate(xs_copy,0.f)/sum));
   };

   auto prec_predicted = [&max_idx,&net]( auto const& pairs )
   {
      auto correct_prediction = [&](auto const& t){ return max_idx(t.second) == max_idx(net(t.first)); };
      return float(rng::count_if(pairs, correct_prediction)) / float(pairs.size());
   };

   std::cout << "• correct predictions within learned set: " << 100.f * prec_predicted(training_data) << "%" << std::endl;
/*
   for (size_t n=0; n<10; ++n)
   {
      auto& p = training_data[n];
      std::cout << "---------------------" << std::endl;
      //for (auto x : p.first) std::cout << x << ", "; std::cout << std::endl;
      for (auto x : p.second) std::cout << x << ", "; std::cout << std::endl;
      for (auto x : net(p.first)) std::cout << x << ", "; std::cout << std::endl;
      std::cout << mean_idx(net(p.first)) << std::endl;
   }
*/

   training_data = [&]
   {
      std::vector<TrainingPair>  training_data;

      for (size_t n=state_size; n < test_signal.size(); ++n)
         training_data.push_back({ {test_signal.data()+n-state_size,state_size}, prob(test_signal[n]) });

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
         pred_signal[n] = float(mean_idx(net({ pred_signal.data()+n-state_size,state_size})))/num_bins;
      for (size_t n=0; n<signal.size(); ++n)
         data_f << signal[n] << " " << pred_signal[n] << std::endl;
   }




}



