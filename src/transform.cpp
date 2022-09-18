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
      std::ofstream outfile0,outfile1, outfile2;
      outfile0.open("O_1.txt");
      outfile1.open("O_n1.txt");
      outfile2.open("O_n2.txt");
      //Fire up the devices.
      sycl::buffer<int> a{10};
      auto fut1 = dpl::experimental::fill_async(policy,
                                                  dpl::begin(a),dpl::end(a),7);
      auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [&](const int& x){return x + 1; },fut1);
        //Start testing transform_async with lambda function of time complexity O1.
        for(int j = 0; j < 256; j++){
          int test_size = 1024 * (j + 1);
          sycl::buffer<int> a{test_size};
          auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);
          fut1.wait();
          auto start = std::chrono::system_clock::now();

          auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [&](const int& x){return x + 1;},fut1);
          fut2.wait();
          auto stop = std::chrono::system_clock::now();  
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);  
          outfile0 << duration.count() <<std::endl;    
        }
        outfile0.close();
        //Start testing transform_async with lambda function of time complexity O(n).
        for(int j = 0; j < 256; j++){
          int test_size = 1024 * (j + 1);
          sycl::buffer<int> a{test_size};
          auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);
          fut1.wait();
          auto start = std::chrono::system_clock::now();

          auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [j](const int& x){int k = 0;for(int i = 0; i < j; i++){ k = k + 1;} return k;},fut1);
          fut2.wait();
          auto stop = std::chrono::system_clock::now();  
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);  
          outfile1 << duration.count() <<std::endl;    
        }
        outfile1.close();
        //Start testing transform_async with lambda function of time complexity O(n2).
        for(int j = 0; j < 256; j++){
          int test_size = 1024 * (j + 1);
          sycl::buffer<int> a{test_size};
          auto fut1 = dpl::experimental::fill_async(dpl::execution::dpcpp_default,
                                                  dpl::begin(a),dpl::end(a),7);
          fut1.wait();
          auto start = std::chrono::system_clock::now();

          auto fut2 = dpl::experimental::transform_async(dpl::execution::dpcpp_default,
                                                       dpl::begin(a),dpl::end(a),dpl::begin(a),
                                                       [j](const int& x){int k = 0;
                                                                                for(int i = 0; i < j; i++){ 
                                                                                    for(int l = 0; l < j; l++){
                                                                                        k += 1;
                                                                                    }
                                                                                } 
                                                                                return k;
                                                                                },fut1);
          fut2.wait();
          auto stop = std::chrono::system_clock::now();  
          auto duration = duration_cast<std::chrono::microseconds>(stop - start);  
          outfile2 << duration.count() <<std::endl;    
        }
        outfile2.close();
        //Start testing transform_async with lambda function of time complexity O(n3).
        
        
    }
    return 0;
}