#ifndef		_NETWORK_HPP_
#define		_NETWORK_HPP_

#include	<list>
using std::list;

#include	<vector>
using std::vector;


#include	<tr1/tuple>
using std::tr1::tuple;
using std::tr1::get;



template <
   class  node_class
>
struct network {

   typedef   tuple < size_t , double >   weighted_link;

   // the Node-class extends the given class with a list of links
   struct node : public node_class {
      list < weighted_link >  in;
   };

private:
   vector<node>   _node;

public:
   network()  { }
   network( size_t N_node ) : _node( N_node ) { }
  ~network()  { }

   typedef  typename vector<node>::iterator  iterator;


   node&  operator[] ( size_t n )  { return  _node[n];         }

   vector<node> &  nodee()  { return _node; }

//   list<node>::iterator   begin()   { return  _node.begin();    }
//   list<node>::iterator   end()     { return  _node.end();      }

   ///////////////////////////////////////////////////////////////

   void  setChainTopology ()
   {
      _node[0].in.clear();
      _node[0].in.push_back( weighted_link(1,1.0) );
      for ( int n=1; n<_node.size()-1; ++n ) {
         _node[n].in.clear();
         _node[n].in.push_back( weighted_link(n-1,1.0) );
         _node[n].in.push_back( weighted_link(n+1,1.0) );
      }
      _node[_node.size()-1].in.clear();
      _node[_node.size()-1].in.push_back( weighted_link(_node.size()-2,1.0) );
   }

   void  set_1D_open_laplacian ()
   {
      setChainTopology();
      _node[0].in.push_back( weighted_link(0,-1.0) );
      for ( int n=1; n<_node.size()-1; ++n )
         _node[n].in.push_back( weighted_link(n,-2.0) );
      _node[_node.size()-1].in.push_back( weighted_link(_node.size()-1,-1.0) );
   }

   void  setRingTopology ()
   {
      _node[0].in.clear();
      _node[0].in.push_back( weighted_link(_node.size()-1,1.0) );
      _node[0].in.push_back( weighted_link(1,1.0) );
      for ( int n=1; n<_node.size()-1; ++n ) {
         _node[n].in.clear();
         _node[n].in.push_back( weighted_link(n-1,1.0) );
         _node[n].in.push_back( weighted_link(n+1,1.0) );
      }
      _node[_node.size()-1].in.clear();
      _node[_node.size()-1].in.push_back( weighted_link(_node.size()-2,1.0) );
      _node[_node.size()-1].in.push_back( weighted_link(0,1.0) );
   }


   void   setGridTopology ( int DX , int DY )
   {
//    int side_len = (int)sqrt( (float)_node.size() );
      if ( DX*DY > _node.size() )  return;

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
//    int side_len = (int)sqrt( (float)_node.size() );
    if ( DX*DY > _node.size() )  return;

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

