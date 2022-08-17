#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <oneapi/dpl/async>
#include <CL/sycl.hpp>
#include <chrono>
int main(int argc, char *argv[]) {
    auto start = std::chrono::system_clock::now();
    //For postfixing each run result;
    //std::time_t time = std::chrono::system_clock::to_time_t(start);
    std::string postfix = "";//ctime(&time);
    /*std::replace(postfix.begin(), postfix.end(), ' ', '-');
    postfix = postfix.substr(0, postfix.size()-1);
    std::cout<<"postfix: "<<postfix ;*/
    //Start testing
    using namespace oneapi;
    {
        auto policy = oneapi::dpl::execution::dpcpp_default;
        std::cout << "Run on "
              << policy.queue().get_device().template
                                        get_info<sycl::info::device::name>()
              << std::endl;
        std::ofstream outfile;

        outfile.open("fill_async.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
            sycl::buffer<int> a{test_size};
            auto start = std::chrono::high_resolution_clock::now();
            auto fut1 = dpl::experimental::fill_async(policy,dpl::begin(a),dpl::end(a),(1<<31));
            fut1.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count() <<std::endl;
        }
        outfile.close();

        outfile.open("reduce_async.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
            sycl::buffer<int> a{test_size};
            auto fut1 = dpl::experimental::fill_async(policy,dpl::begin(a),dpl::end(a),(1<<31));
            fut1.wait();
            auto start = std::chrono::high_resolution_clock::now();
            auto ret_val = dpl::experimental::reduce_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),fut1).get();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count() <<std::endl;
        }
        outfile.close();

        outfile.open("sort_async.txt_"+postfix);
        for(int j = 1; j < 1024; j++){ //caviat: sort must start with buff size at least 2
            int test_size = 1024 * (j + 1);
            sycl::buffer<int> a{test_size};
            auto fut1 = dpl::experimental::fill_async(policy,dpl::begin(a),dpl::end(a),(1<<31));
            fut1.wait();
            auto start = std::chrono::high_resolution_clock::now();
            auto ret_val = dpl::experimental::sort_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),fut1);
            ret_val.wait();
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count() <<std::endl;
        }
        
        outfile.close();
        
        outfile.open("inclusive_scan_async.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
                    sycl::buffer<int> a{test_size};
                    sycl::buffer<int> b{test_size};
                    auto fut1 = dpl::experimental::fill_async(policy,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::inclusive_scan_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),dpl::begin(b), std::plus<>{}, fut1);
                    ret_val.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count() <<std::endl;
        }
        outfile.close();

        outfile.open("copy_async.txt_"+postfix);
         for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
                    //use addition
                    sycl::buffer<int> a{test_size};
                    sycl::buffer<int> b{test_size};
                    auto fut1 = dpl::experimental::fill_async(policy,dpl::begin(a),dpl::end(a),(1<<31));
                    fut1.wait();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::experimental::copy_async(dpl::execution::dpcpp_default,
                                                            dpl::begin(a),dpl::end(a),dpl::begin(b), fut1);
                    ret_val.wait();
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count() <<std::endl;
        }
        outfile.close();

        //oneapi::transform
        //oneapi::stable_sort
        //oneapi::for_each
        
        outfile.open("max_element.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
                    sycl::buffer<int> a{test_size};
                    std::fill(policy,dpl::begin(a), dpl::end(a), 42);
                    auto start = std::chrono::high_resolution_clock::now();
                    auto ret_val = dpl::max_element(policy,dpl::begin(a),dpl::end(a));
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count()<<std::endl;
        }
        outfile.close();

        outfile.open("distance.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int test_size = 1024 * (j + 1);
                    sycl::buffer<int> a{test_size};
                    std::fill(policy,dpl::begin(a), dpl::end(a), 42);
                    auto maxloc = dpl::max_element(policy,dpl::begin(a),dpl::end(a));
                    auto start = std::chrono::high_resolution_clock::now();
                    auto result = dpl::distance(oneapi::dpl::begin(a), maxloc);
                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
            outfile <<duration.count()<<std::endl;
        }
        outfile.close();  

        std::ofstream outfile2;
        outfile.open("transform.txt_"+postfix);
        outfile2.open("stable_sort.txt_"+postfix);
        for(int j = 0; j < 1024; j++){
            int n = 1024 * (j + 1);
                    sycl::buffer<int> keys_buf{n};  // buffer with keys
                    sycl::buffer<int> vals_buf{n};  // buffer with values

                    // create objects to iterate over buffers
                    auto keys_begin = oneapi::dpl::begin(keys_buf);
                    auto vals_begin = oneapi::dpl::begin(vals_buf);
                    auto counting_begin = oneapi::dpl::counting_iterator<int>{0};
                    // use default policy for algorithms execution

                    // 1. Initialization of buffers
                    // let keys_buf contain {n, n, n-2, n-2, ..., 4, 4, 2, 2}
                    ///////////////////////////////API////////////////////////////////////////
                    auto start = std::chrono::high_resolution_clock::now();
                    transform(policy, counting_begin, counting_begin + n, keys_begin,
                                [n](int i) { return n - (i / 2) * 2; });

                    auto stop = std::chrono::high_resolution_clock::now();
                    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
                    outfile <<duration.count()<<std::endl;
                    //////////////////////////////////////////////////////////////////////////
                    // fill vals_buf with the analogue of std::iota using counting_iterator
                    std::copy(policy, counting_begin, counting_begin + n, vals_begin);
                    // 2. Sorting
                    auto zipped_begin = oneapi::dpl::make_zip_iterator(keys_begin, vals_begin);
                    
                    // stable sort by keys
                    ///////////////////////////////API////////////////////////////////////////
                    start = std::chrono::high_resolution_clock::now();
                    dpl::stable_sort(
                        policy, zipped_begin, zipped_begin + n,
                        // Generic lambda is needed because type of lhs and rhs is unspecified.
                        [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });
                                    
                                    
                    stop = std::chrono::high_resolution_clock::now();
                    duration = duration_cast<std::chrono::microseconds>(stop - start);
                    //////////////////////////////////////////////////////////////////////////
        outfile2 <<duration.count()<<std::endl;
    }
    outfile.close();
    outfile2.close();
    return 0;
    }
}