/*
 * Created by Maximiliano Levi on 4/12/2021.
 */
#include <chrono>

long long TimeIt(std::chrono::time_point<std::chrono::steady_clock>& prev_time)
{
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - prev_time ).count();
    prev_time = t2;
    return duration;
}