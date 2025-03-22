#include <iostream>
#include <thread>

void printNaturalNumbers(int n)
{
    for (int i = 1; i <= n; ++i)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main()
{
    int n;
    std::cout << "Enter the value of n: ";
    std::cin >> n;

    std::thread t(printNaturalNumbers, n);
    t.join();

    return 0;
}