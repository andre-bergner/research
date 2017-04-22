#pragma once

#include "terminal.hpp"

#include <iostream>
#include <sstream>


namespace terminal {

namespace cursor {

   std::ostream& operator<<( std::ostream& os, absolute_pos pos )
   {
      return  os << "\033[" << (pos.row+1) << ";" << (pos.column+1) << "H";
   }

   std::ostream& operator<<( std::ostream& os, relative_pos pos )
   {
      if (pos.row > 0)      os << "\033[" <<  pos.row << "B";
      else if (pos.row < 0) os << "\033[" << -pos.row << "A";

      if (pos.column > 0)      os << "\033[" <<  pos.column << "C";
      else if (pos.column < 0) os << "\033[" << -pos.column << "D";
    
      return  os;
   }

   std::ostream& operator<<( std::ostream& os, hide_t )
   {
      return  os << "\033[?25l";
   }

   std::ostream& operator<<( std::ostream& os, show_t )
   {
      return  os << "\033[?25h";
   }
}

namespace font {

   std::ostream& operator<<( std::ostream& os, color c )
   {
      return os << "\033[" << (30 + static_cast<int>(c)) << 'm';
   }

   std::ostream& operator<<( std::ostream& os, background c )
   {
      return os << "\033[" << (40 + static_cast<int>(c)) << 'm';
   }

   std::ostream& operator<<( std::ostream& os, weight w )
   {
      return os << "\033[" << (w == weight::bold ? "1" : "21;") << 'm';
   }

   std::ostream& operator<<( std::ostream& os, reset_t )
   {
      return os << "\033[0m";
   }

}


std::ostream& operator<<( std::ostream& os, erase e )
{
   static const auto codes = {"2J", "1J", "0J", "2K", "1K", "0K"};
   return  os << "\033[" << codes.begin()[ static_cast<size_t>(e) ];
}


}



#include <termios.h>
#include <unistd.h>
#include <fcntl.h>

namespace terminal {
namespace cursor {


   // https://www.gnu.org/software/libc/manual/html_node/Noncanonical-Input.html#Noncanonical-Input
   // http://pubs.opengroup.org/onlinepubs/7908799/xsh/termios.h.html

   class noncanon_silent_terminal
   {
      ::termios  prev_terminal_state;
   public:
      noncanon_silent_terminal()
      {
         ::tcgetattr(STDIN_FILENO, &prev_terminal_state);

         ::termios new_state = prev_terminal_state;
         new_state.c_lflag &= not ICANON;  // so we can read each input instead by line
         new_state.c_lflag &= not ECHO;    // input should not be printed
      
         ::tcsetattr(STDIN_FILENO, TCSANOW, &new_state);
      }
      
      ~noncanon_silent_terminal()
      {
         ::tcsetattr(STDIN_FILENO, TCSANOW, &prev_terminal_state);
      }
   };

   absolute_pos current_position()
   {
      noncanon_silent_terminal noncan_term;

      std::cin.sync();
      std::cout << "\033[6n" << std::flush;

      absolute_pos p = {0,0};
      bool reading_col = false;
      char c;

      auto getch = []
      {
          unsigned char c;
          if ( ::read(0, &c, sizeof(c)) < 0 ) throw;
          return c;
      };


      fcntl( STDIN_FILENO, F_SETFL, O_NONBLOCK );
      // getchar()     // FIXME Not thread safe, deadlock if other thread cout's.
      // Potential solution:
      // https://groups.google.com/forum/#!topic/comp.os.linux.development.apps/XwDVOIqTsQ0
      // http://stackoverflow.com/questions/448944/c-non-blocking-keyboard-input

      do
      {
         c = ::getchar();
         if (c == '\033' || c == '[')
            continue;
         else if (c == ';')
            reading_col = true;
         else if ('0' <= c && c <= '9')
         {
            if (reading_col == false)
               p.row = ((p.row * 10) + (c - '0'));
            else
               p.column = ((p.column * 10) + (c - '0'));
         }
      } while (c != 'R');

      // we are using 0 based indices
      --p.row;
      --p.column;

      return p;
   }

}}

