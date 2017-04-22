#include <cstddef>

template <typename T>
struct context_pool_allocator
{
   using value_type = T;

   context_pool_allocator() {};

   value_type* allocate(std::size_t n)
   {
      //auto pool& = get_pool_for_id(...);
      return pool.allocate( n * sizeof(T) );
   }

   void deallocate(value_type* p, std::size_t n);

   // static hash< context_id → stack<context_pool> >

}

template <typename T, typename U>
bool operator==(const context_pool_allocator<T>&, const context_pool_allocator<U>&);

template <typename T, typename U>
bool operator!=(const context_pool_allocator<T>&, const context_pool_allocator<U>&);



// solving the scope-context problem
// * ids could be thread ids
//   → would not work with custom coroutine implementations as used in boost.context v2
// * user can pass lambda that delegates to custom id-provider
//   e.g.  context_pool p(1024, []{ return this_thread::id(); } );
//   or    context_pool p(1024, []{ return my_context::id(); } );
//   or    context_pool p(1024, my_context::id );  // just passing the id function should work as well.



#include <vector>
#include <functional>

class context_pool
{
   using id_func = std::function< int() >;
   using memory_t = std::vector<unsigned char>;
   
   memory_t  bytes;
   size_t    pool_head = 0;
   size_t    number_of_allocs = 0;

   id_func   idf;

public:

   void* allocate( size_t num_bytes )
   {
      pool_head += num_bytes;
      assert( pool_head < bytes.size() );
      ++number_of_allocs;
      return  bytes.data() + pool_head;
   }

   void* deallocate(size_t)
   {
      assert( number_of_allocs > 0 );
      if ( --number_of_allocs == 0 )  pool_head = 0;   // free the pool at once
   }

   context_pool( size_t num_bytes, id_func const& f = []{return 0;} )
   :  bytes(num_bytes)
   ,  idf(f)
   {}

   context_pool( context_pool const& ) = delete;   // cannot copy resources
   context_pool( context_pool&& ) = default;
   
   context_pool& operator=( context_pool const& ) = delete;   // cannot copy resources
   context_pool& operator=( context_pool&& ) = default;

};




#include <iostream>


int main()
{
   template <typename T>
   using vec_t = std::vector<T,context_pool_allocator<T>>;


   context_pool p( 1024*size(int) );      // construct a pool that can hold 1024 floats
   {
      vec_t<int> v1(128,0);               // pool1 capacity used: 128
      vec_t<int> v2(128,1337);            // pool1 capacity used: 256
      {
         vec_t<int> v3(256,42);           // pool1 capacity used: 512
         vec_t<int> v4(256,47);           // pool1 capacity used: 768
      }                                   // pool1 capacity used: 768
      {
         vec_t<int> v5(64,1);             // pool1 capacity used: 832
         vec_t<int> v6(64,2);             // pool1 capacity used: 896
      }                                   // pool1 capacity used: 896

      context_pool q( 128*size(int) );    // 2nd pool that can hold 128 ints
      {
         vec_t<int> u1(64,1);             // pool2 capacity used: 64
         vec_t<int> u2(64,3);             // pool2 capacity used: 128  -- full
      }                                   // pool2 emptied

   }  // pool2 destroyed,   pool1 capacity used: 0,   everything gets released here


}
