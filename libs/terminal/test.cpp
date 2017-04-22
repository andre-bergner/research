#include "terminal_shop"

#include <iostream>

int main()
{
   using namespace std;
   using namespace terminal_shop;

   cout << erase::screen        << endl;

   cout << cursor::absolute_pos{0,0} << "XXXXXXXX" << endl;
   cout << cursor::absolute_pos{10,10} << "X";
   cout << cursor::relative_pos{0,1}  << "X";
   cout << cursor::relative_pos{0,1}  << "X";
   cout << cursor::relative_pos{0,1}  << "X";
   cout << cursor::relative_pos{0,1}  << "X";
   cout << cursor::hide;
   cout << cursor::show;
   cout << endl;
   cout << endl;
   cout << endl;

   using namespace font;

   for ( auto b : { background::white, background::red, background::green, background::yellow,
                    background::blue, background::magenta, background::cyan, background::black } )
   {
      cout << b;

      for ( auto c : { color::white, color::red, color::green, color::yellow,
                       color::blue, color::magenta, color::cyan, color::black } )
         cout << c << "•";

      cout << endl;
   }

   cout << color::white << endl;
   cout << string(20,'X') << cursor::relative_pos{-10,0} << erase::line_before << endl;
   cout << string(20,'Y') << cursor::relative_pos{-10,0} << erase::line_after << endl;
   cout << string(20,'Z') << cursor::relative_pos{-10,0} << erase::line << endl;

   //cout << erase::screen_before << endl;
   //cout << erase::screen_after  << endl;

   /*
   cout << font::color::white     << "white"   << endl;
   cout << font::color::red       << "red"     << endl;
   cout << font::color::green     << "green"   << endl;
   cout << font::color::yellow    << "yellow"  << endl;
   cout << font::color::blue      << "blue"    << endl;
   cout << font::color::magenta   << "magenta" << endl;
   cout << font::color::cyan      << "cyan"    << endl;
   cout << font::color::black     << "black"   << endl;

   cout << font::background::white     << "white"   << endl;
   cout << font::background::red       << "red"     << endl;
   cout << font::background::green     << "green"   << endl;
   cout << font::background::yellow    << "yellow"  << endl;
   cout << font::background::blue      << "blue"    << endl;
   cout << font::background::magenta   << "magenta" << endl;
   cout << font::background::cyan      << "cyan"    << endl;
   cout << font::background::black     << "black"   << endl;
   */
   cout << font::weight::bold << "bold" << endl;
   cout << font::weight::normal << "normal" << endl;

   cout << cursor::absolute_pos{22,33} << flush;
   auto p = cursor::current_position();

   cout << p.column << ", " << p.row << endl;

   //cout << "This library sucks!" << Cursor(0, -6, AC_RELATIVE) << "ro" << endl;
   //cout << Cursor(3, 10) << "Look ma', no spaces!" << endl;
}
