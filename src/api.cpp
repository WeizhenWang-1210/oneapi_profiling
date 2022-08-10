#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <chrono>
int main() {
    using namespace oneapi;
    {
      auto policy = oneapi::dpl::execution::dpcpp_default;
      std::cout << "Run on "
            << policy.queue().get_device().template
                                        get_info<sycl::info::device::name>()
            << std::endl;
      std::ofstream outfile;
      outfile.open("combined.txt");
        for(int j = 0; j < 20; j++){
          int test_size = 1<<j;
          sycl::buffer<int> a{test_size};
          auto start = std::chrono::system_clock::now();

          auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);

          auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [&](const int& x){return x + 1; },fut1);
          auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),fut1,fut2).get(); 
          
          auto stop = std::chrono::system_clock::now();  
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);          
        }
        outfile.close();
    }
    return 0;
}