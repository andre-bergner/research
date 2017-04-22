# include <boost/iterator/iterator_facade.hpp>

namespace iter
{
   struct pop_back_iterator_end {};

   template <typename Container>
   class pop_back_iterator : public boost::iterator_facade
                             <  pop_back_iterator<Container>
                             ,  typename Container::value_type
                             ,  boost::forward_traversal_tag
                             >
   {
   public:

      explicit pop_back_iterator(Container& c) : cont_(c) {}

      pop_back_iterator(pop_back_iterator const&) = delete;
      pop_back_iterator(pop_back_iterator&&) = default;

   private:
      friend class boost::iterator_core_access;

      bool equal(pop_back_iterator const& that) const
      {
         return &cont_ == &that.cont_;
      }

      bool equal(pop_back_iterator_end) const
      {
         return not cont_.empty();
      }

      void increment() { cont_.pop_back(); }

      typename Container::value_type& dereference() const { return cont_.back(); }

      Container&  cont_;
   };


   template <typename Container>
   auto  back_remover( Container& c )
   {
      return pop_back_iterator<Container>(c);
   }


}