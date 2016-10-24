#include <atomic>
#include <future>

namespace print
{
   decltype(auto) repeat( std::string s, size_t n )
   {
      auto l = s.size();
      std::string rs(l*n, ' ');
      auto b = rs.begin();
      while ( n --> 0 ) { std::copy_n(s.begin(),l,b); b+=l; }
      return std::move(rs);
   }


   auto make_progress_printer = []( size_t n_max, size_t bar_width = 20, std::string text = "")
   {
      return [n_max, bar_width, text](size_t n)
      {
         auto perc = float(n+1) / float(n_max);
         std::cout << "\r" << text;
         std::cout << repeat( "█", int(bar_width*perc) )
                   << repeat( "░", bar_width-int(bar_width*perc) )
                   << "  ";
         std::cout << n+1 << "/" << n_max << " (" << int(100.f*perc) << "%)";
         std::cout << std::flush;
      };
   };


   class RunningWheel
   {
      std::atomic<bool>  running = {true};
      std::thread        t;
   public:
      RunningWheel()
      : t{[this]
          {
             auto const rotation_char = "|/-\\|";
             int n = 0;
             while (running)
             {
                std::cout << "\r" << rotation_char[n] << std::flush;
                n = ++n & 0b11;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
             }
          }}
      {}

      ~RunningWheel()
      {
         running = false;
         t.join();
         std::cout << "\r✓" << std::endl;
      }
   };


}
