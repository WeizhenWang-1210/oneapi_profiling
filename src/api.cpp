#include <oneapi/dpl/execution>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <chrono>
#include <bits/stdc++.h>
int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();
    //For postfixing each run result;
    std::time_t time = std::chrono::system_clock::to_time_t(start);
    std::string postfix = ctime(&time);
    std::replace(postfix.begin(), postfix.end(), ' ', '-');
    postfix = postfix.substr(0, postfix.size()-1);
    std::cout<<"postfix: "<<postfix ;
    //Start testing
    using namespace oneapi;
    {
        std::ofstream outfile;
        outfile.open("fill_async.txt_"+postfix);
        for(int j = 0; j < 31; j++){
            int test_size = 1<<j;
            for(int i = 0; i < 20; i++){
                {
                    sycl::buffer<int> a{test_size};
                    auto start = std::chrono::high_resolution_clock::now();
                    auto fut1 = dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                }
            }
        }
        outfile.close();
        
        outfile.open("reduce_async.txt"+postfix);
        for(int j = 0; j < 31; j++){
            int test_size = 1<<j;
            for(int i = 0; i < 20; i++){
                {
                    sycl::buffer<int> a{test_size};
                    auto fut1 = dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),fut1).get();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                }
            }
        }
        outfile.close();  
        outfile.open("sort_async.txt"+postfix);
        for(int j = 1; j < 31; j++){ //caviat: sort must start with buff size at least 2
            int test_size = 1<<j;
            for(int i = 0; i < 20; i++){
                {
                    sycl::buffer<int> a{test_size};
                    auto fut1 = dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::sort_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),fut1);
                    ret_val.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                }
            }
        }
        
        outfile.close();
        
        outfile.open("inclusive_scan_async.txt"+postfix);
        for(int j = 0; j < 31; j++){
            int test_size = 1<<j;
            for(int i = 0; i < 20; i++){
                {
                    //use addition
                    sycl::buffer<int> a{test_size};
                    sycl::buffer<int> b{test_size};
                    auto fut1 = dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::inclusive_scan_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),dpl::begin(b), std::plus<>{}, fut1);
                    ret_val.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                }
            }
        }
        outfile.close();
        outfile.open("copy_async.txt"+postfix);
         for(int j = 0; j < 31; j++){
            int test_size = 1<<j;
            for(int i = 0; i < 20; i++){
                {
                    //use addition
                    sycl::buffer<int> a{test_size};
                    sycl::buffer<int> b{test_size};
                    auto fut1 = dpl::experimental::fill_async(oneapi::dpl::execution::dpcpp_default,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::copy_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),dpl::begin(b), fut1);
                    ret_val.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                }
            }    
        }
        outfile.close();
    }
    return 0;
}