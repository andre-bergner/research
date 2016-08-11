#ifndef		_NETWORK_HPP_
#define		_NETWORK_HPP_

#include	<list>
using std::list;

#include	<tr1/tuple>
using std::tr1::tuple;
using std::tr1::get;


template <
   class  node_class,
   int    node_num
>
struct const_network {

   typedef   tuple < size_t , double >   weighted_link;

   // the Node-class extends the given class with a list of links
   struct node : public node_class {
      list < weighted_link >  in;
   };

private:

//   Node     _node [ node_num ];
//  the following four lines are a workaround for the vectorization bug in icc 11.0
//  the above line should be the correct choice ( direct array implementation )
   node   *_node;
public:
   const_network()  { _node = new node[ node_num ]; }
  ~const_network()  { delete [] _node; }

public:

   node&  operator[] ( size_t n )  { return  _node[n];         }
   node*  begin() const            { return  _node;            }
   node*  end()   const            { return  _node + node_num; }

   ///////////////////////////////////////////////////////////////

   void  setChainTopology ()
   {
      _node[0].in.clear();
      _node[0].in.push_back( weighted_link(1,1.0) );
      for ( int n=1; n<node_num-1; ++n ) {
         _node[n].in.clear();
         _node[n].in.push_back( weighted_link(n-1,1.0) );
         _node[n].in.push_back( weighted_link(n+1,1.0) );
      }
      _node[node_num-1].in.clear();
      _node[node_num-1].in.push_back( weighted_link(node_num-2,1.0) );
   }

   void  set_1D_open_laplacian ()
   {
      setChainTopology();
      _node[0].in.push_back( weighted_link(0,-1.0) );
      for ( int n=1; n<node_num-1; ++n )
         _node[n].in.push_back( weighted_link(n,-2.0) );
      _node[node_num-1].in.push_back( weighted_link(node_num-1,-1.0) );
   }

   void  setRingTopology ()
   {
      _node[0].in.clear();
      _node[0].in.push_back( weighted_link(node_num-1,1.0) );
      _node[0].in.push_back( weighted_link(1,1.0) );
      for ( int n=1; n<node_num-1; ++n ) {
         _node[n].in.clear();
         _node[n].in.push_back( weighted_link(n-1,1.0) );
         _node[n].in.push_back( weighted_link(n+1,1.0) );
      }
      _node[node_num-1].in.clear();
      _node[node_num-1].in.push_back( weighted_link(node_num-2,1.0) );
      _node[node_num-1].in.push_back( weighted_link(0,1.0) );
   }


   void   setGridTopology ( int DX , int DY )
   {
//    int side_len = (int)sqrt( (float)node_num );
      if ( DX*DY > node_num )  return;

      for ( int n=0; n<DY; ++n )
         for ( int m=0; m<DX; ++m ) {
            _node[n*DX+m].in.clear();
            if ( n > 0 )      _node[n*DX+m].in.push_back( weighted_link((n-1)*DX+m,1.0) );
            if ( n < DX-1 )   _node[n*DX+m].in.push_back( weighted_link((n+1)*DX+m,1.0) );
            if ( m > 0 )      _node[n*DX+m].in.push_back( weighted_link(n*DX+m-1,1.0) );
            if ( m < DX-1 )   _node[n*DX+m].in.push_back( weighted_link(n*DX+m+1,1.0) );
         }
   }



  void   setPeriodicGridTopology ( int DX , int DY )
  {
//    int side_len = (int)sqrt( (float)node_num );
    if ( DX*DY > node_num )  return;

    for ( int n=0; n<DY; ++n )
      for ( int m=0; m<DX; ++m ) {
        _node[n*DX+m].in.clear();
        _node[n*DX+m].in.push_back( weighted_link(((n-1<0)?DX-1:n-1)*DX+m,1.0) );
        _node[n*DX+m].in.push_back( weighted_link(((n+1>=DX)?0:n+1)*DX+m,1.0) );
        _node[n*DX+m].in.push_back( weighted_link(n*DX+((m-1<0)?DY-1:m-1),1.0) );
        _node[n*DX+m].in.push_back( weighted_link(n*DX+((m+1>=DY)?0:m+1),1.0) );
      }
  }

};

#endif

