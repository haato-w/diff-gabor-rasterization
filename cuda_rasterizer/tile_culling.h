/*
 * Perspective surfel tile culling and warp-cooperative tile enumeration.
 * The support is the UNION of the projected surfel disk and the renderer's
 * screen-space low-pass disk. Integer pixel centers match renderCUDA.
 */
#pragma once

struct TileSupport
{
    float2 center;
    float2 filter_center;
    float3 conic;
    float filter_radius2;
};
static_assert(sizeof(TileSupport) == 32, "Tile support scratch must stay 32 bytes per primitive");

__forceinline__ __device__ TileSupport unrestricted_tile_support()
{
    TileSupport support = {};
    support.conic.x = -1.0f;
    return support;
}

__forceinline__ __device__ TileSupport make_tile_support(
    const glm::mat3& T, const float opacity, const float2 filter_center)
{
    TileSupport support = unrestricted_tile_support();
    support.filter_center = filter_center;
    if (!isfinite(opacity) || !isfinite(filter_center.x) || !isfinite(filter_center.y))
        return support;
    if (opacity < 1.0f / 255.0f)
    {
        support.conic.x = 0.0f; // No sample reaches the alpha threshold.
        return support;
    }
    float r2 = 2.0f * logf(opacity * 255.0f);
    if (!isfinite(r2)) return support;
    // Expand the analytic support to cover the renderer's float arithmetic.
    r2 = fmaxf(r2, 0.0f) + 1.0e-3f + 1.0e-4f * fmaxf(r2, 0.0f);
    support.filter_radius2 = r2 / FilterInvSquare + 1.0e-3f;

    const glm::vec3 a = T[0], b = T[1], w = T[2];
    const float wl2 = fmaf(w.x, w.x, w.y * w.y);
    const float wz2 = w.z * w.z;
    const float denominator = fmaf(-r2, wl2, wz2);
    // An ellipse is finite only while the local disk stays on one side of the
    // projective horizon. Retain every candidate tile in unstable cases.
    if (!isfinite(denominator) || !isfinite(wz2) ||
        !(denominator > 1.0e-3f * wz2)) return support;
    support.center = make_float2(
        fmaf(-r2, fmaf(a.x, w.x, a.y * w.y), a.z * w.z) / denominator,
        fmaf(-r2, fmaf(b.x, w.x, b.y * w.y), b.z * w.z) / denominator);

    // Factor the projected ellipse covariance without subtracting large
    // center-squared terms. The determinant is the squared factor cross product.
    float ax, ay, bx, by;
    if (wl2 > 0.0f)
    {
        const float wl = sqrtf(wl2);
        const float factor = sqrtf(r2) / denominator;
        const float perpendicular_factor = factor * sqrtf(denominator) / wl;
        ax = factor * fmaf(fmaf(a.x, w.x, a.y * w.y) / wl, w.z, -a.z * wl);
        bx = factor * fmaf(fmaf(b.x, w.x, b.y * w.y) / wl, w.z, -b.z * wl);
        ay = perpendicular_factor * fmaf(a.x, w.y, -a.y * w.x);
        by = perpendicular_factor * fmaf(b.x, w.y, -b.y * w.x);
    }
    else
    {
        const float factor = sqrtf(r2) / fabsf(w.z);
        ax = factor * a.x;
        ay = factor * a.y;
        bx = factor * b.x;
        by = factor * b.y;
    }
    const float xx = fmaf(ax, ax, ay * ay);
    const float xy = fmaf(ax, bx, ay * by);
    const float yy = fmaf(bx, bx, by * by);
    const float cross = fmaf(ax, by, -ay * bx);
    const float determinant = cross * cross;
    if (!isfinite(determinant) || !(xx > 0.0f) || !(yy > 0.0f) ||
        !(determinant > 1.0e-8f * xx * yy) ||
        !isfinite(support.center.x) || !isfinite(support.center.y)) return unrestricted_tile_support();
    support.conic = make_float3(yy / determinant, -xy / determinant, xx / determinant);
    if (!isfinite(support.conic.x) || !isfinite(support.conic.y) || !isfinite(support.conic.z))
        return unrestricted_tile_support();
    return support;
}

__forceinline__ __device__ bool tile_may_contribute(
    const TileSupport& support, const uint32_t tile_x, const uint32_t tile_y)
{
    if (support.conic.x < 0.0f) return true;
    if (support.conic.x == 0.0f) return false;
    const float xmin = static_cast<float>(tile_x * BLOCK_X);
    const float ymin = static_cast<float>(tile_y * BLOCK_Y);
    const float xmax = xmin + BLOCK_X - 1;
    const float ymax = ymin + BLOCK_Y - 1;

    const float filter_dx = fmaxf(fmaxf(xmin - support.filter_center.x, 0.0f), support.filter_center.x - xmax);
    const float filter_dy = fmaxf(fmaxf(ymin - support.filter_center.y, 0.0f), support.filter_center.y - ymax);
    if (fmaf(filter_dx, filter_dx, filter_dy * filter_dy) <= support.filter_radius2)
        return true;

    const float2 c = support.center;
    const float3 q = support.conic;
    const bool outside_x = c.x < xmin || c.x > xmax;
    const bool outside_y = c.y < ymin || c.y > ymax;
    if (!outside_x && !outside_y) return true;

    // Minimize the positive quadratic on the box. Start at the corner facing
    // the mean and slide along its two incident edges to their stationary point.
    // For an SPD conic at most one slide can leave this corner when both axes
    // are outside; this also covers a mean outside only one axis.
    const float corner_x = c.x <= xmin ? xmin : xmax;
    const float corner_y = c.y <= ymin ? ymin : ymax;
    const float direction_x = c.x <= xmin ? BLOCK_X - 1.0f : 1.0f - BLOCK_X;
    const float direction_y = c.y <= ymin ? BLOCK_Y - 1.0f : 1.0f - BLOCK_Y;
    const float diff_x = c.x - corner_x;
    const float diff_y = c.y - corner_y;
    const float tx = outside_y ? __saturatef(fmaf(q.x, diff_x, q.y * diff_y) / (direction_x * q.x)) : 0.0f;
    const float ty = outside_x ? __saturatef(fmaf(q.y, diff_x, q.z * diff_y) / (direction_y * q.z)) : 0.0f;
    const float dx = c.x - fmaf(tx, direction_x, corner_x);
    const float dy = c.y - fmaf(ty, direction_y, corner_y);
    const float x_term = q.x * dx * dx;
    const float cross_term = 2.0f * q.y * dx * dy;
    const float y_term = q.z * dy * dy;
    const float minimum = fmaf(q.x * dx, dx, fmaf(2.0f * q.y * dx, dy, y_term));
    const float guard = 1.0e-3f + 1.0e-5f * (fabsf(x_term) + fabsf(cross_term) + fabsf(y_term));
    return !isfinite(minimum) || minimum <= 1.0f + guard;
}

__forceinline__ __device__ TileSupport shuffle_tile_support(const TileSupport& s, const int source)
{
    TileSupport r;
    r.center = make_float2(__shfl_sync(0xffffffff, s.center.x, source), __shfl_sync(0xffffffff, s.center.y, source));
    r.conic = make_float3(__shfl_sync(0xffffffff, s.conic.x, source), __shfl_sync(0xffffffff, s.conic.y, source), __shfl_sync(0xffffffff, s.conic.z, source));
    r.filter_center = make_float2(__shfl_sync(0xffffffff, s.filter_center.x, source), __shfl_sync(0xffffffff, s.filter_center.y, source));
    r.filter_radius2 = __shfl_sync(0xffffffff, s.filter_radius2, source);
    return r;
}

__forceinline__ __device__ void emit_tile_instance(
    const uint32_t primitive, const float depth, const uint32_t tile_x,
    const uint32_t tile_y, const uint32_t grid_width, const uint32_t offset,
    uint64_t* keys, uint32_t* values)
{
    const uint64_t key = (static_cast<uint64_t>(tile_y * grid_width + tile_x) << 32) | __float_as_uint(depth);
    keys[offset] = key;
    values[offset] = primitive;
}

// The SAME predicate and row-major traversal serve count and emission. Full
// warps enter this function, including out-of-range or invisible lanes.
template<bool Emit>
__forceinline__ __device__ uint32_t enumerate_tiles(
    const uint32_t primitive, const uint4 bounds, const TileSupport support,
    const float depth, const uint32_t offset, const uint32_t grid_width,
    const bool active, const bool tile_culling, const bool load_balance,
    const int sequential_threshold, uint64_t* keys = nullptr, uint32_t* values = nullptr)
{
    const uint32_t width = bounds.z - bounds.x;
    const uint32_t total = active ? width * (bounds.w - bounds.y) : 0;
    // Counting unrestricted/empty support does not need any tile tests. Keep
    // these lanes in the warp so they can still assist heavy neighbors.
    const bool direct_count = !Emit && (!tile_culling || support.conic.x <= 0.0f);
    const uint32_t enumeration_total = direct_count ? 0 : total;
    const uint32_t sequential = load_balance ? min(enumeration_total, static_cast<uint32_t>(sequential_threshold)) : enumeration_total;
    uint32_t count = direct_count && (!tile_culling || support.conic.x < 0.0f) ? total : 0;
    for (uint32_t i = 0; i < sequential; ++i)
    {
        const uint32_t x = bounds.x + i % width;
        const uint32_t y = bounds.y + i / width;
        if (!tile_culling || tile_may_contribute(support, x, y))
        {
            if (Emit) emit_tile_instance(primitive, depth, x, y, grid_width, offset + count, keys, values);
            ++count;
        }
    }
    if (!load_balance) return count;
    const uint32_t lane = threadIdx.x & 31;
    uint32_t heavy = __ballot_sync(0xffffffff, enumeration_total > sequential);
    while (heavy)
    {
        const int owner = __ffs(heavy) - 1;
        const uint32_t owner_primitive = __shfl_sync(0xffffffff, primitive, owner);
        const uint32_t xmin = __shfl_sync(0xffffffff, bounds.x, owner);
        const uint32_t ymin = __shfl_sync(0xffffffff, bounds.y, owner);
        const uint32_t owner_width = __shfl_sync(0xffffffff, width, owner);
        const uint32_t owner_total = __shfl_sync(0xffffffff, enumeration_total, owner);
        const uint32_t owner_sequential = __shfl_sync(0xffffffff, sequential, owner);
        const float owner_depth = __shfl_sync(0xffffffff, depth, owner);
        uint32_t write_offset = __shfl_sync(0xffffffff, offset + count, owner);
        const TileSupport owner_support = shuffle_tile_support(support, owner);
        for (uint32_t start = owner_sequential; start < owner_total; start += 32)
        {
            const uint32_t i = start + lane;
            const uint32_t x = xmin + i % owner_width;
            const uint32_t y = ymin + i / owner_width;
            const bool keep = i < owner_total && (!tile_culling || tile_may_contribute(owner_support, x, y));
            const uint32_t ballot = __ballot_sync(0xffffffff, keep);
            if (Emit && keep)
            {
                const uint32_t previous_lanes = (1u << lane) - 1u;
                emit_tile_instance(owner_primitive, owner_depth, x, y, grid_width,
                    write_offset + __popc(ballot & previous_lanes), keys, values);
            }
            const uint32_t retained = __popc(ballot);
            if (lane == owner) count += retained;
            write_offset += retained;
        }
        heavy &= heavy - 1;
    }
    return count;
}
