#pragma once
#include <algorithm>
#include <numeric>


namespace rng
{
   template <typename It> struct range_t { It b,e; };

   template <typename It> It begin(range_t<It>& r)       { return r.b; }
   template <typename It> It end(range_t<It>& r)         { return r.e; }
   template <typename It> It begin(range_t<It> const& r) { return r.b; }
   template <typename It> It end(range_t<It> const& r)   { return r.e; }

   template <typename It> range_t<It> range(It b, It e) { return {b,e}; }

   
   template <typename Range, typename Iterator>
   decltype(auto) copy( Range const& r, Iterator i )
   {
      return std::copy( std::begin(r), std::end(r), i );
   }

   template <typename Range, typename Function>
   decltype(auto) generate( Range& r, Function&& f )
   {
      return std::generate( std::begin(r), std::end(r), std::forward<Function>(f) );
   }

   template <typename Range, typename Value>
   decltype(auto) fill( Range& r, Value&& v )
   {
      return std::fill( std::begin(r), std::end(r), std::forward<Value>(v) );
   }

   template <typename Range, typename Function>
   decltype(auto) for_each( Range&& r, Function&& f )
   {
      return std::for_each( std::begin(r), std::end(r), std::forward<Function>(f) );
   }

   template <typename Range1, typename Range2, typename Func>
   decltype(auto) transform( Range1 const& r1, Range2& r2, Func&& f )
   {
      return std::transform( std::begin(r1), std::end(r1), std::begin(r2), std::forward<Func>(f) );
   }

   template <typename Range1, typename Range2, typename Range3, typename Func>
   decltype(auto) transform( Range1 const& r1, Range2 const& r2, Range3& r3, Func&& f )
   {
      return std::transform( std::begin(r1), std::end(r1), std::begin(r2), std::begin(r3), std::forward<Func>(f) );
   }

   template <typename Range, typename Number>
   decltype(auto) iota( Range& r, Number&& init )
   {
      return std::iota( std::begin(r), std::end(r), std::forward<Number>(init) );
   }

   template <typename Range, typename Func>
   decltype(auto) shuffle( Range& r, Func&& f )
   {
      return std::shuffle( std::begin(r), std::end(r), std::forward<Func>(f) );
   }

   template <typename Range, typename Func>
   decltype(auto) count_if( Range& r, Func&& f )
   {
      return std::count_if( std::begin(r), std::end(r), std::forward<Func>(f) );
   }

/*
   template <typename Func, typename Value, typename... Ranges>
   auto fold( Func&& f, Value&& Range1 const& r, Range2 const& r2, Range3& r3, )
   {
      return std::transform( std::begin(r1), std::end(r1), std::begin(r2), std::begin(r3), std::forward<Func>(f) );
   }
*/


   struct reverse_t {};

   template <typename Range>
   auto operator| (Range&& r, reverse_t) { return range(r.rbegin(),r.rend()); }

   namespace { auto&& reverse = reverse_t{}; }
}



namespace stat
{

   template <typename Value>
   class Averager
   {
      Value        value_ = {};
      std::size_t  count_ = {};
   public:
      decltype(auto) operator()(Value&& x)
      {
         value_ += x;
         ++count_;
         return std::forward<Value>(x);
      }

      Value get() const { return value_ / static_cast<Value>(count_); }
   };

}


