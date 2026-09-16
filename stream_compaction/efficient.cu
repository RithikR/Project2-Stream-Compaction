#include <cuda.h>
#include <cuda_runtime.h>

#include "common.h"
#include "efficient.h"

namespace StreamCompaction {

    namespace Efficient {

        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int dataIndex = (index + 1) * offset * 2 - 1;

            if (dataIndex < n) {
                data[dataIndex] += data[dataIndex - offset];
            }
        }

        __global__ void kernDownSweep(int n, int offset, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int dataIndex = (index + 1) * offset * 2 - 1;

            if (dataIndex < n) {
                int temp = data[dataIndex - offset];
                data[dataIndex - offset] = data[dataIndex];
                data[dataIndex] += temp;
            }
        }

        __global__ void kernSetZero(int n, int* data) {
            if (blockIdx.x == 0 && threadIdx.x == 0) {
                data[n - 1] = 0;
            }
        }

        static void scanDevice(int n, int* data) {
            const int blockSize = 256;
            int numThreads = n / 2;
            int numBlocks = (numThreads + blockSize - 1) / blockSize;

            // Up-sweep
            for (int offset = 1; offset < n; offset *= 2) {
                kernUpSweep << <numBlocks, blockSize >> > (
                    n,
                    offset,
                    data
                    );
            }

            // Clear root for exclusive scan
            kernSetZero << <1, 1 >> > (
                n,
                data
                );

            // Down-sweep
            for (int offset = n / 2; offset >= 1; offset /= 2) {
                kernDownSweep << <numBlocks, blockSize >> > (
                    n,
                    offset,
                    data
                    );
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return;
            }

            int paddedN = 1 << ilog2ceil(n);
            int* dev_data;

            cudaMalloc(
                (void**)&dev_data,
                paddedN * sizeof(int)
            );

            cudaMemset(
                dev_data,
                0,
                paddedN * sizeof(int)
            );

            cudaMemcpy(
                dev_data,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            timer().startGpuTimer();

            scanDevice(
                paddedN,
                dev_data
            );

            timer().endGpuTimer();

            cudaMemcpy(
                odata,
                dev_data,
                n * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_data);

            checkCUDAError("Work-efficient scan failed");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int* odata, const int* idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedN = 1 << ilog2ceil(n);

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc(
                (void**)&dev_idata,
                n * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_odata,
                n * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_bools,
                paddedN * sizeof(int)
            );

            cudaMalloc(
                (void**)&dev_indices,
                paddedN * sizeof(int)
            );

            cudaMemcpy(
                dev_idata,
                idata,
                n * sizeof(int),
                cudaMemcpyHostToDevice
            );

            cudaMemset(
                dev_bools,
                0,
                paddedN * sizeof(int)
            );

            cudaMemset(
                dev_indices,
                0,
                paddedN * sizeof(int)
            );

            const int blockSize = 256;
            const int numBlocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Map
            StreamCompaction::Common::kernMapToBoolean << <numBlocks, blockSize >> > (
                n,
                dev_bools,
                dev_idata
                );

            cudaMemcpy(
                dev_indices,
                dev_bools,
                paddedN * sizeof(int),
                cudaMemcpyDeviceToDevice
            );

            // Scan
            scanDevice(
                paddedN,
                dev_indices
            );

            // Scatter
            StreamCompaction::Common::kernScatter << <numBlocks, blockSize >> > (
                n,
                dev_odata,
                dev_idata,
                dev_bools,
                dev_indices
                );

            timer().endGpuTimer();

            int lastIndex;
            int lastBool;

            cudaMemcpy(
                &lastIndex,
                dev_indices + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaMemcpy(
                &lastBool,
                dev_bools + n - 1,
                sizeof(int),
                cudaMemcpyDeviceToHost
            );

            int count = lastIndex + lastBool;

            cudaMemcpy(
                odata,
                dev_odata,
                count * sizeof(int),
                cudaMemcpyDeviceToHost
            );

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            checkCUDAError("Work-efficient compaction failed");

            return count;
        }

    }

}