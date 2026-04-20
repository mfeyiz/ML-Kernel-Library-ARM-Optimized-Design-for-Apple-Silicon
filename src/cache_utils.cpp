#include "gemm.h"
#include <sys/sysctl.h>
#include <cstddef>

namespace hwml {

size_t get_cache_size() {
    size_t cache_size = 65536;
    size_t len = sizeof(cache_size);
    sysctlbyname("hw.l2cachesize", &cache_size, &len, NULL, 0);
    if (cache_size == 0) cache_size = 65536;
    return cache_size;
}

}
