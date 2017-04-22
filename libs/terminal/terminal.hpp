#pragma once

#include <iosfwd>

namespace terminal
{


   // FUNCTIONS ----------------------------------------------------------------

   namespace cursor
   {
      struct relative_pos;
      struct absolute_pos;
      struct hide_t;
      struct show_t;

      absolute_pos current_position();

      std::ostream& operator<<( std::ostream&, relative_pos );
      std::ostream& operator<<( std::ostream&, absolute_pos );
      std::ostream& operator<<( std::ostream&, hide_t );
      std::ostream& operator<<( std::ostream&, show_t );
   }


   namespace font
   {
      enum class color;
      enum class background;
      enum class weight;
      struct reset_t;

      std::ostream& operator<<( std::ostream&, color );
      std::ostream& operator<<( std::ostream&, background );
      std::ostream& operator<<( std::ostream&, weight );
      std::ostream& operator<<( std::ostream&, reset_t );
   }


   enum class erase
   {  screen
   ,  screen_before
   ,  screen_after
   ,  line
   ,  line_before
   ,  line_after
   };

   std::ostream& operator<<( std::ostream& os, erase e );


   // TYPES --------------------------------------------------------------------

   namespace cursor
   {
      struct relative_pos
      {
         int column;
         int row;
      };

      struct absolute_pos
      {
         unsigned column;
         unsigned row;
      };

      struct hide_t {};
      struct show_t {};

      namespace
      {
         hide_t hide;
         show_t show;
      }
   }



   namespace font
   {
      enum class color
      {  black
      ,  red
      ,  green
      ,  yellow
      ,  blue
      ,  magenta
      ,  cyan
      ,  white
      };


      enum class background
      {  black
      ,  red
      ,  green
      ,  yellow
      ,  blue
      ,  magenta
      ,  cyan
      ,  white
      };


      enum class weight
      {  normal
      ,  bold
      };

      struct reset_t {};
      namespace { reset_t reset; }
   }
}


#ifdef TERMINAL_HEADER_ONLY

#  define TERMINAL_ANSI_SUPPORT     // FIXME: figure out from settings

#  ifdef TERMINAL_ANSI_SUPPORT
#     include "terminal.impl.ansi.hpp"
#  endif

#  ifdef TERMINAL_WINDOWS_API
#     include "terminal.impl.win.hpp"
#  endif

#endif
