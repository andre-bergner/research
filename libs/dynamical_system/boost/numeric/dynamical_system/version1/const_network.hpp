//
//
//
//

#ifndef		_NETWORK_
#define		_NETWORK_

#include	<vector>

  class Connection {
  public:
    int    id;
    float  weight;
    float  delay;

    Connection()     // standard c'tor
    : id(0) , weight(0.0) , delay(0.0)
    { }

    Connection( const int i , const float w=1.0 , const float d=0.0 )
    : id(i) , weight(w) , delay(d)
    { }

//    Connection( const int i , const float w , const float d )   { id = i;   weight = w;  delay = d; }
//    Connection( const int i , const float w )                   { id = i;   weight = w;  delay = 0.0; }

/*
    void     set_weight  ( const float val )     { _weight = val;  }
    void     setWeight   ( const float val )     { _weight = val;  }
    void     SetWeight   ( const float val )     { _weight = val;  }
    float &  weight()                            { return _weight; }

    void     set_id  ( const int val )   { _id = val; }
    void     setId   ( const int val )   { _id = val; }
    void     SetId   ( const int val )   { _id = val; }
    int &    id()                        { return _id; }
*/
  };



template< class _node_base , int nodeNum >
class const_network
{
private:

  struct Node : public _node_base {
    std::vector<Connection>  in, out;
  };

//   Node     _node [ nodeNum ];
//  the following four lines are a workaround for the vectorization bug in icc 11.0
//  the above line should be the correct choice ( direct array implementation )
  Node   *_node;
public:
  const_network() { _node = new Node[ nodeNum ]; }
 ~const_network() { delete [] _node; }


public:

  std::vector<Connection> &   in_node  ( int n )  { return _node[n].in;  }
  std::vector<Connection> &   out_node ( int n )  { return _node[n].out; }
  Node &                      node     ( int n )  { return _node[n]; }

   Node &  operator[] ( size_t n )  { return _node[n]; }



  void   setChainTopology ()
  {
    _node[0].in.clear();
    _node[0].in.push_back( Connection(1,1.0) );
    for ( int n=1; n<nodeNum-1; ++n ) {
      _node[n].in.clear();
      _node[n].in.push_back( Connection(n-1,1.0) );
      _node[n].in.push_back( Connection(n+1,1.0) );
    }
    _node[nodeNum-1].in.clear();
    _node[nodeNum-1].in.push_back( Connection(nodeNum-2,1.0) );
  }

  void   set_1D_open_laplacian ()
  {
    setChainTopology();
    _node[0].in.push_back( Connection(0,-1.0) );
    for ( int n=1; n<nodeNum-1; ++n )
      _node[n].in.push_back( Connection(n,-2.0) );
    _node[nodeNum-1].in.push_back( Connection(nodeNum-1,-1.0) );
  }

  void   setRingTopology ()
  {
    _node[0].in.clear();
    _node[0].in.push_back( Connection(nodeNum-1,1.0) );
    _node[0].in.push_back( Connection(1,1.0) );
    for ( int n=1; n<nodeNum-1; ++n ) {
      _node[n].in.clear();
      _node[n].in.push_back( Connection(n-1,1.0) );
      _node[n].in.push_back( Connection(n+1,1.0) );
    }
    _node[nodeNum-1].in.clear();
    _node[nodeNum-1].in.push_back( Connection(nodeNum-2,1.0) );
    _node[nodeNum-1].in.push_back( Connection(0,1.0) );
  }


  void   setGridTopology ( int DX , int DY )
  {
//    int side_len = (int)sqrt( (float)nodeNum );
    if ( DX*DY > nodeNum )  return;

    for ( int n=0; n<DY; ++n )
      for ( int m=0; m<DX; ++m ) {
        _node[n*DX+m].in.clear();
       if ( n > 0 )      _node[n*DX+m].in.push_back( Connection((n-1)*DX+m,1.0) );
       if ( n < DX-1 )   _node[n*DX+m].in.push_back( Connection((n+1)*DX+m,1.0) );
       if ( m > 0 )      _node[n*DX+m].in.push_back( Connection(n*DX+m-1,1.0) );
       if ( m < DX-1 )   _node[n*DX+m].in.push_back( Connection(n*DX+m+1,1.0) );
      }
  }


  void   setTwoLayerGridTopology ( int DX , int DY )
  {
//    int side_len = (int)sqrt( (float)nodeNum );
    if ( 2*DX*DY > nodeNum )  return;

    for ( int n=0; n<DY; ++n )
      for ( int m=0; m<DX; ++m ) {
        _node[2*(n*DX+m)].in.clear();
        _node[2*(n*DX+m)+1].in.clear();
        if ( n > 0 )    { _node[2*(n*DX+m)].in.push_back( Connection(2*((n-1)*DX+m),1.0) );
                          _node[2*(n*DX+m)+1].in.push_back( Connection(2*((n-1)*DX+m)+1,1.0) ); }
        if ( n < DX-1 ) { _node[2*(n*DX+m)].in.push_back( Connection(2*((n+1)*DX+m),1.0) );
                          _node[2*(n*DX+m)+1].in.push_back( Connection(2*((n+1)*DX+m)+1,1.0) ); }
        if ( m > 0 )    { _node[2*(n*DX+m)].in.push_back( Connection(2*(n*DX+m-1),1.0) );
                          _node[2*(n*DX+m)+1].in.push_back( Connection(2*(n*DX+m-1)+1,1.0) );   }
        if ( m < DX-1 ) { _node[2*(n*DX+m)].in.push_back( Connection(2*(n*DX+m+1),1.0) );
                          _node[2*(n*DX+m)+1].in.push_back( Connection(2*(n*DX+m+1)+1,1.0) );   }
        _node[2*(n*DX+m)].in.push_back( Connection(2*(n*DX+m)+1,  -0.1 ) );
        _node[2*(n*DX+m)+1].in.push_back( Connection(2*(n*DX+m),  -0.5 ) );
      }
  }



  void   setPeriodicGridTopology ( int DX , int DY )
  {
//    int side_len = (int)sqrt( (float)nodeNum );
    if ( DX*DY > nodeNum )  return;

    for ( int n=0; n<DY; ++n )
      for ( int m=0; m<DX; ++m ) {
        _node[n*DX+m].in.clear();
        _node[n*DX+m].in.push_back( Connection(((n-1<0)?DX-1:n-1)*DX+m,1.0) );
        _node[n*DX+m].in.push_back( Connection(((n+1>=DX)?0:n+1)*DX+m,1.0) );
        _node[n*DX+m].in.push_back( Connection(n*DX+((m-1<0)?DY-1:m-1),1.0) );
        _node[n*DX+m].in.push_back( Connection(n*DX+((m+1>=DY)?0:m+1),1.0) );
      }
  }

};



#endif


