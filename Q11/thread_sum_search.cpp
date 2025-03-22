// #include <iostream>
// #include <vector>
// #include <cstdlib>
// #include <ctime>
// #include <thread> // Add this include
// #include <chrono>

// using namespace std;

// // Function to generate a random integer array
// vector<int> generateArray(int n)
// {
//     vector<int> arr(n);
//     for (int i = 0; i < n; ++i)
//     {
//         arr[i] = rand() % 1000;
//     }
//     return arr;
// }

// // Function to find the sum of elements in an array
// int findSum(const vector<int> &arr)
// {
//     int sum = 0;
//     for (int num : arr)
//     {
//         sum += num;
//     }
//     return sum;
// }

// // Function to search for a key element in an array
// bool searchKey(const vector<int> &arr, int key)
// {
//     for (int num : arr)
//     {
//         if (num == key)
//             return true;
//     }
//     return false;
// }

// // Thread function for finding sum in partitioned arrays
// void threadedSum(const vector<int> &arr, int start, int end, int &result)
// {
//     result = 0;
//     for (int i = start; i < end; ++i)
//     {
//         result += arr[i];
//     }
// }

// // Thread function for searching key in partitioned arrays
// void threadedSearch(const vector<int> &arr, int start, int end, int key, bool &found)
// {
//     found = false;
//     for (int i = start; i < end; ++i)
//     {
//         if (arr[i] == key)
//         {
//             found = true;
//             break;
//         }
//     }
// }

// int main()
// {
//     srand(time(0));
//     int n = 100000; // Array size
//     vector<int> arr = generateArray(n);
//     int key = arr[n / 2]; // Select a random key from the array

//     // Sequential Execution
//     auto start = chrono::high_resolution_clock::now();
//     int sum = findSum(arr);
//     bool found = searchKey(arr, key);
//     auto end = chrono::high_resolution_clock::now();
//     cout << "Sequential Sum: " << sum << " Execution Time: " << chrono::duration<double>(end - start).count() << "s\n";
//     cout << "Sequential Search Found: " << found << "\n";

//     // Multithreading Execution
//     int mid = n / 2;
//     int sum1, sum2;
//     bool found1, found2;
//     thread t1(threadedSum, ref(arr), 0, mid, ref(sum1));
//     thread t2(threadedSum, ref(arr), mid, n, ref(sum2));
//     t1.join();
//     t2.join();
//     int totalSum = sum1 + sum2;

//     thread t3(threadedSearch, ref(arr), 0, mid, key, ref(found1));
//     thread t4(threadedSearch, ref(arr), mid, n, key, ref(found2));
//     t3.join();
//     t4.join();
//     bool keyFound = found1 || found2;

//     cout << "Multithreading Sum: " << totalSum << "\n";
//     cout << "Multithreading Search Found: " << keyFound << "\n";

//     return 0;
// }

#include <iostream>
#include <thread>

void hello()
{
    std::cout << "Hello from thread!\n";
}

int main()
{
    std::thread t(hello);
    t.join();
    return 0;
}
