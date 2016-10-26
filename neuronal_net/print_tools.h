#define TERMINAL_HEADER_ONLY
#include <terminal/terminal.hpp>

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
      terminal::cursor::absolute_pos  wheel_pos;
      std::atomic<bool>  running = {true};
      std::thread        t;
   public:
      RunningWheel( terminal::cursor::absolute_pos wp = terminal::cursor::current_position() )
      : wheel_pos(wp)
      , t{[this]
          {
             auto const rotation_char = "|/-\\|";
             int n = 0;
             while (running)
             {
                auto cur_pos = terminal::cursor::current_position();
                std::cout << wheel_pos << rotation_char[n] << cur_pos << std::flush;
                n = ++n & 0b11;
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
             }
          }}
      {}

      ~RunningWheel()
      {
         using namespace terminal::font;
         running = false;
         t.join();
         auto cur_pos = terminal::cursor::current_position();
         std::cout << wheel_pos << color::green << weight::bold << "✓" << reset << cur_pos << std::flush;
      }
   };


}
