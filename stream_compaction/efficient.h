#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanDevice(int n, int* data);

        void scan(int n, int* odata, const int* idata);

        int compact(int n, int* odata, const int* idata);
    }
}