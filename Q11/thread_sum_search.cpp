#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <thread>
#include <atomic>

using namespace std;
using namespace chrono;

// Function to generate a random integer array
vector<int> generateArray(int n)
{
    vector<int> arr(n);
    for (int i = 0; i < n; i++)
    {
        arr[i] = i;
    }
    return arr;
}

// ** Sequential Functions **

// Function to find the sum of elements in an array
long long findSumSequential(const vector<int> &arr)
{
    long long sum = 0;
    for (int num : arr)
    {
        sum += num;
    }
    return sum;
}

// Function to search for a key in an array
bool searchKeySequential(const vector<int> &arr, int key)
{
    for (int num : arr)
    {
        if (num == key)
        {
            return true;
        }
    }
    return false;
}

// ** Multi-threaded Functions **

// Thread function to compute sum of a partition
void sumPartition(const vector<int> &arr, int start, int end, long long &partialSum)
{
    partialSum = 0;
    for (int i = start; i < end; i++)
    {
        partialSum += arr[i];
    }
}

// Thread function to search for a key in a partition
void searchPartition(const vector<int> &arr, int start, int end, int key, atomic<bool> &found)
{
    for (int i = start; i < end; i++)
    {
        if (arr[i] == key)
        {
            found = true;
            return;
        }
    }
}

int main()
{
    srand(time(0)); // Seed random number generator
    int n = 0;
    int key = 0;
    cout << "ENTER THE SIZE OF ARRAY : ";
    cin >> n;
    cout << "ENTER THE KEY FOR SEARCH : ";
    cin >> key;
    int numThreads = 4; // Number of threads

    // Generate array
    vector<int> arr = generateArray(n);

    // ** Sequential Execution **
    cout << "\n--- Sequential Execution ---\n";

    // Sum computation
    auto start = high_resolution_clock::now();
    long long sumSeq = findSumSequential(arr);
    auto end = high_resolution_clock::now();
    cout << "Sequential Sum: " << sumSeq << endl;
    cout << "Time taken (Sequential Sum): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    // Key search
    start = high_resolution_clock::now();
    bool foundSeq = searchKeySequential(arr, key);
    end = high_resolution_clock::now();
    cout << "Key " << key << (foundSeq ? " found" : " not found") << " in Sequential Search." << endl;
    cout << "Time taken (Sequential Search): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    // ** Multi-threaded Execution **
    cout << "\n--- Multi-threaded Execution ---\n";

    // Multi-threaded sum computation
    vector<thread> threads;
    vector<long long> partialSums(numThreads, 0);
    int chunkSize = n / numThreads;

    start = high_resolution_clock::now();
    for (int i = 0; i < numThreads; i++)
    {
        int startIdx = i * chunkSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + chunkSize;
        threads.push_back(thread(sumPartition, ref(arr), startIdx, endIdx, ref(partialSums[i])));
    }

    for (auto &t : threads)
    {
        t.join();
    }

    long long totalSum = 0;
    for (long long partSum : partialSums)
    {
        totalSum += partSum;
    }

    end = high_resolution_clock::now();
    cout << "Multi-threaded Sum: " << totalSum << endl;
    cout << "Time taken (Multi-threaded Sum): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    // Multi-threaded key search
    vector<atomic<bool>> found(numThreads); // Use atomic<bool> to avoid bitwise issues with vector<bool>
    threads.clear();
    start = high_resolution_clock::now();

    for (int i = 0; i < numThreads; i++)
    {
        int startIdx = i * chunkSize;
        int endIdx = (i == numThreads - 1) ? n : startIdx + chunkSize;
        threads.push_back(thread([&arr, startIdx, endIdx, key, &found, i]()
                                 { searchPartition(arr, startIdx, endIdx, key, found[i]); }));
    }

    for (auto &t : threads)
    {
        t.join();
    }

    bool keyFound = false;
    for (int i = 0; i < numThreads; i++)
    {
        if (found[i])
        {
            keyFound = true;
            break;
        }
    }

    end = high_resolution_clock::now();
    cout << "Key " << key << (keyFound ? " found" : " not found") << " in Multi-threaded Search." << endl;
    cout << "Time taken (Multi-threaded Search): " << duration_cast<milliseconds>(end - start).count() << " ms" << endl;

    return 0;
}