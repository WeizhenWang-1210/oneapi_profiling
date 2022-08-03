#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
#include <CL/sycl.hpp>
#include <chrono>
int main(){
  sycl::buffer<int> buf { 1000 };
  auto buf_begin = oneapi::dpl::begin(buf);
  auto buf_end   = oneapi::dpl::end(buf);
  std::fill(oneapi::dpl::execution::dpcpp_default, buf_begin, buf_end, 42);
  return 0;
}

//
        //oneapi::dpl::max_element
        //oneapi::dpl::distance
        //oneapi::transform
        //oneapi::stable_sort
        //oneapi::dpl::make_zip_iterator
        //oneapi::for_each
