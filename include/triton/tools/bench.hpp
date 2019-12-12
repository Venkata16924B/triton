#pragma once

#ifndef _TRITON_TOOLS_BENCH_H_
#define _TRITON_TOOLS_BENCH_H_

#include <chrono>
#include <functional>
#include <algorithm>
#include "triton/driver/device.h"
#include "triton/driver/stream.h"

namespace triton{
namespace tools{

class timer{
    typedef std::chrono::high_resolution_clock high_resolution_clock;
    typedef std::chrono::nanoseconds nanoseconds;

public:
    explicit timer(bool run = false)
    { if (run) start(); }

    void start()
    { _start = high_resolution_clock::now(); }

    nanoseconds get() const
    { return std::chrono::duration_cast<nanoseconds>(high_resolution_clock::now() - _start); }

private:
    high_resolution_clock::time_point _start;
};

inline double bench(std::function<void()> const & op, driver::stream * stream, bool normalize = true)
{
    timer tmr;
    op();
    stream->synchronize();
    // estimate number of repeats so that the total time stays under .1s
    size_t repeat = 0;
    std::vector<size_t> times;
    double total_time = 0;
    while(total_time*1e-9 < 1e-1){
      float norm = 1;
      // normalize clock if possible to reduce noise in auto-tuning
      if(auto cu_device = dynamic_cast<const triton::driver::cu_device*>(stream->context()->device()))
        norm = (float)cu_device->current_sm_clock()/cu_device->max_sm_clock();
      tmr.start();
      op();
      stream->synchronize();
      total_time+=tmr.get().count();
      if(normalize)
        times.push_back(norm*tmr.get().count());
      repeat += 1;
    }
    if(normalize)
      return *std::min_element(times.begin(), times.end());
    // enqueues a batch of task to make sure it runs at high clock speed
    tmr.start();
    for(size_t i = 0; i < repeat; i++)
      op();
    stream->synchronize();
    return (double)tmr.get().count() / repeat;
}

}
}

#endif
