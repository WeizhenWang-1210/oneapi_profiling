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
      std::ofstream outfile, outfile2;
      outfile.open("combined.txt");
      outfile2.open("Individual.txt");
      sycl::buffer<int> a{10};
      auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);
      auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),fut1,fut1).get(); 
        for(int j = 0; j < 1024; j++){
          int test_size = 1024 * (j + 1);
          sycl::buffer<int> a{test_size};
          auto start = std::chrono::system_clock::now();
          auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);
          auto breakpoint1 = std::chrono::system_clock::now();
          auto duration1 = duration_cast<std::chrono::microseconds>(breakpoint1 - start);  
          //

          auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),fut1).get(); 
          
          auto stop = std::chrono::system_clock::now();  
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);  
          auto duration2 =   duration_cast<std::chrono::microseconds>(stop - breakpoint1);  
          outfile << duration.count() <<std::endl;
          outfile2 << duration1.count()<<";"<<duration2.count()<<std::endl;     
        }
        outfile.close();
    }
    return 0;
}