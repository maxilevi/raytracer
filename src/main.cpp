#include <iostream>
#include "Vector3.h"

int main()
{
    /* TODO handle simulation and create a png from the result */
    Vector3 vec1(1, 2, 3);
    Vector3 vec2(1, 2, 3);

    auto vec3 = vec1 + vec2;
    std::cout << "Hello, World!" << "\n" << vec3 << std::endl;
    return 0;
}