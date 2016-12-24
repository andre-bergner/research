#include <vector>
#include <array>
#include <algorithm>
#include <cassert>



template <typename Value>
class matrix
{
   std::size_t         n_rows_;
   std::size_t         n_cols_;
   std::vector<Value>  data_;

public:

   matrix() = default;

   matrix( std::size_t n_rows, std::size_t n_cols, Value const& x = {} )
   : n_rows_ (n_rows)
   , n_cols_ (n_cols)
   , data_   (n_cols*n_rows, x)
   {}

   matrix& resize( std::size_t n_rows, std::size_t n_cols )
   {
      n_rows_ = n_rows;
      n_cols_ = n_cols;
      data_.resize( n_cols * n_rows );
      return *this;
   }

   matrix& operator=( std::initializer_list<Value> const& values )
   {
      std::copy( values.begin(), values.end(), data_.begin() );
      return *this;
   }

   auto& operator()( std::size_t m , std::size_t n )        { return data_[ m + n_rows_*n ]; }
   auto  operator()( std::size_t m , std::size_t n ) const  { return data_[ m + n_rows_*n ]; }
//   auto& operator()( std::size_t m , std::size_t n )        { return data_.at(m + n_rows_*n); }
//   auto  operator()( std::size_t m , std::size_t n ) const  { return data_.at(m + n_rows_*n); }

   size_t row_size() const { return n_rows_; }
   size_t col_size() const { return n_cols_; }

   decltype(auto) begin()       { return data_.begin(); }
   decltype(auto) end()         { return data_.end(); }

   decltype(auto) begin() const { return data_.begin(); }
   decltype(auto) end()   const { return data_.end(); }

   decltype(auto) data()       { return data_.data(); }
   decltype(auto) data() const { return data_.data(); }
   
   decltype(auto) size() const { return data_.size(); }
};


template <typename Value>
class Span
{
   Value*       data_;
   std::size_t  size_;

public:

   Span( Value* d, std::size_t s ) : data_{d} , size_{s} {}

   Span( std::vector<Value>& v ) : data_{v.data()} , size_{v.size()} {}
   Span( std::vector<std::decay_t<Value>> const& v ) : data_{v.data()} , size_{v.size()} {}

   template <size_t N>
   Span( std::array<Value,N>& v ) : data_{v.data()} , size_{v.size()} {}
   template <size_t N>
   Span( std::array<std::decay_t<Value>,N> const& v ) : data_{v.data()} , size_{v.size()} {}


   //Span( Span const& ) = default;
   //Span( Span<std::remove_const_t<Value>> const& s ) : data_{s.data()} , size_{s.size()} {}

   Span( Span<std::remove_const_t<Value>> const& s ) : data_{s.data()} , size_{s.size()} {}



   Value* data() const  { return data_; }
   Value* begin() const { return data_; }
   Value* end() const   { return data_ + size_; }

   std::size_t size() const  { return size_; }

   Value& operator[]( std::size_t n ) const { return data_[n]; }
   Value& at( std::size_t n ) const { if (n >= size()) throw; return data_[n]; }
};



template <typename Vector1, typename Vector2, typename Vector3>
void add( Vector1 const& u, Vector2 const& v, Vector3& w )
{
   for ( size_t k = 0 ; k < std::min(u.size(),v.size()) ; ++k )
      w[k] = u[k] + v[k];
}


template <typename Matrix, typename Vector1, typename Vector2>
void dot_add( Matrix const& m, Vector1 const& a, Vector2& b )
{
   for ( size_t l = 0 ; l < m.col_size() ; ++l )
      for ( size_t k = 0 ; k < m.row_size() ; ++k )
         b[k] += m(k,l)*a[l];
}


template <typename Matrix, typename Vector1, typename Vector2>
void dot( Matrix const& m, Vector1 const& a, Vector2& b )
{
   for ( size_t k = 0 ; k < m.row_size() ; ++k ) b[k] = 0.f;
   dot_add(m,a,b);
}



template <typename Matrix, typename Vector1, typename Vector2>
void dott_add( Matrix const& m, Vector1 const& a, Vector2& b )
{
   for ( size_t l = 0 ; l < m.col_size() ; ++l )
      for ( size_t k = 0 ; k < m.row_size() ; ++k )
         b[l] += m(k,l)*a[k];
}


template <typename Matrix, typename Vector1, typename Vector2>
void dott( Matrix const& m, Vector1 const& a, Vector2& b )
{
   for ( size_t l = 0 ; l < m.col_size() ; ++l ) b[l] = 0.f;
   dott_add(m,a,b);
}

template <typename Matrix, typename Vector>
auto dott( Matrix const& m, Vector const& a )
{
   Vector b(m.col_size(),0.f);
   dott_add(m,a,b);
   return b;
}





// assumes that left vector is row and right is column
template <typename Vector, typename Matrix>
void outer( Vector const& v, Vector const& u, Matrix& m )
{
   assert( m.col_size() == u.size() );
   assert( m.row_size() == v.size() );

   for ( size_t l = 0 ; l < u.size() ; ++l )
      for ( size_t k = 0 ; k < v.size() ; ++k )
         m(k,l) = v[k]*u[l];
}

template <typename Vector>
auto outer( Vector const& u, Vector const& v )
{
   matrix<typename Vector::value_type> m(v.size(),u.size());
   outer(u,v,m);
   return m;
}


