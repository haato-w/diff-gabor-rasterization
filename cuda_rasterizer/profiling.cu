#include "profiling.h"

#include <cuda_runtime_api.h>
#include <atomic>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace CudaRasterizer {
namespace Profiling {

namespace {
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Rasterizer profiling: ") + cudaGetErrorString(status));
}

class DeviceGuard {
public:
    explicit DeviceGuard(int device)
    {
        check(cudaGetDevice(&previous_));
        if (previous_ != device)
            check(cudaSetDevice(device));
    }
    ~DeviceGuard() noexcept { cudaSetDevice(previous_); }

private:
    int previous_;
};
} // namespace

struct Record {
    explicit Record(const char* stage) : name(stage) {}
    ~Record() noexcept
    {
        // Events also get cleaned up if a kernel wrapper or collection throws.
        int previous = -1;
        cudaGetDevice(&previous);
        if (device >= 0 && previous != device)
            cudaSetDevice(device);
        if (start) cudaEventDestroy(start);
        if (end) cudaEventDestroy(end);
        if (previous >= 0 && previous != device)
            cudaSetDevice(previous);
    }

    std::string name;
    int device = -1;
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
};

namespace {
struct State {
    std::atomic<bool> enabled{false};
    std::mutex mutex;
    std::vector<std::unique_ptr<Record>> pending;
    std::exception_ptr error;
};

State& state()
{
    // Shared across Python and CUDA autograd worker threads.
    static State value;
    return value;
}
} // namespace

void setEnabled(bool enabled)
{
    state().enabled.store(enabled, std::memory_order_relaxed);
}

Scope::Scope(const char* name)
{
    if (!state().enabled.load(std::memory_order_relaxed))
        return;
    std::unique_ptr<Record> record(new Record(name));
    check(cudaGetDevice(&record->device));
    check(cudaEventCreate(&record->start));
    check(cudaEventCreate(&record->end));
    // The existing rasterizer kernels and CUB calls use the CUDA default stream.
    check(cudaEventRecord(record->start, nullptr));
    pending_ = record.release();
}

Scope::~Scope() noexcept { stop(); }

void Scope::stop() noexcept
{
    if (!pending_)
        return;
    std::unique_ptr<Record> record(pending_);
    pending_ = nullptr;
    try {
        std::lock_guard<std::mutex> lock(state().mutex);
        // Recording and insertion share a lock, preserving end-event stream order.
        check(cudaEventRecord(record->end, nullptr));
        state().pending.emplace_back(std::move(record));
    } catch (...) {
        // A destructor must not replace an in-flight kernel exception.
        try {
            std::lock_guard<std::mutex> lock(state().mutex);
            if (!state().error)
                state().error = std::current_exception();
        } catch (...) {}
    }
}

std::map<std::string, double> collect()
{
    std::vector<std::unique_ptr<Record>> records;
    std::exception_ptr error;
    {
        std::lock_guard<std::mutex> lock(state().mutex);
        records.swap(state().pending);
        error.swap(state().error);
    }
    if (error)
        std::rethrow_exception(error);

    // Waiting once on the final end event per device is enough for all intervals.
    // An iteration profiler that already synchronized incurs no extra wait here.
    std::map<int, Record*> last;
    for (const auto& record : records)
        last[record->device] = record.get();
    for (const auto& entry : last) {
        DeviceGuard device(entry.first);
        cudaError_t status = cudaEventQuery(entry.second->end);
        if (status == cudaErrorNotReady)
            check(cudaEventSynchronize(entry.second->end));
        else
            check(status);
    }

    std::map<std::string, double> totals;
    for (const auto& record : records) {
        DeviceGuard device(record->device);
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, record->start, record->end));
        totals[record->name] += milliseconds;
    }
    return totals;
}

} // namespace Profiling
} // namespace CudaRasterizer
