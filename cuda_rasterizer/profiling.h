#pragma once

#include <map>
#include <string>

namespace CudaRasterizer {
namespace Profiling {

// Rebuild the extension after changing these CUDA stage markers.
// Timings are default-stream CUDA event intervals in milliseconds, not CPU timers.
void setEnabled(bool enabled);
std::map<std::string, double> collect();

struct Record;

class Scope {
public:
    explicit Scope(const char* name);
    ~Scope() noexcept;
    void stop() noexcept;

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Record* pending_ = nullptr;
};

} // namespace Profiling
} // namespace CudaRasterizer
