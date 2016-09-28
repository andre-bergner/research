#include <algorithm>


namespace rng
{
   
   template <typename Range, typename Iterator>
   auto copy( Range const& r, Iterator i )
   {
      return std::copy( std::begin(r), std::end(r), i );
   }

   template <typename Range, typename Function>
   auto generate( Range& r, Function&& f )
   {
      return std::generate( std::begin(r), std::end(r), std::forward<Function>(f) );
   }

   template <typename Range, typename Value>
   auto fill( Range& r, Value&& v )
   {
      return std::fill( std::begin(r), std::end(r), std::forward<Value>(v) );
   }

   template <typename Range1, typename Range2, typename Range3, typename Func>
   auto transform( Range1 const& r1, Range2 const& r2, Range3& r3, Func&& f )
   {
      return std::transform( std::begin(r1), std::end(r1), std::begin(r2), std::begin(r3), std::forward<Func>(f) );
   }

/*
   template <typename Func, typename Value, typename... Ranges>
   auto fold( Func&& f, Value&& Range1 const& r, Range2 const& r2, Range3& r3, )
   {
      return std::transform( std::begin(r1), std::end(r1), std::begin(r2), std::begin(r3), std::forward<Func>(f) );
   }
*/
}