#  include  <string>
#  include  <boost/date_time/posix_time/ptime.hpp>
#  include  <boost/date_time/microsec_time_clock.hpp>


struct Timer
{
   Timer ( const std::string & name )
   :  _name ( name ),
      _start( boost::date_time::microsec_clock<boost::posix_time::ptime>::local_time() )
   { }
   
   ~Timer() {
      using namespace std;
      using namespace boost;
      
      posix_time::ptime
         now ( date_time::microsec_clock<posix_time::ptime>::local_time() );

      posix_time::time_duration
         d = now - _start;
      
      cout << _name << " completed in " << d.total_milliseconds() / 1000.0 <<
      " seconds" << endl;
   }


private:

   std::string               _name;
   boost::posix_time::ptime  _start;

};
