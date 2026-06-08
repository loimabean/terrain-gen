// Combined optimization: LUT gradient lookup + all-octave shared memory cache.
//
// Eliminates all sin/cos (like compute_lut.wgsl) AND amortizes hash+LUT lookups
// across threads by caching every octave's gradient corners in workgroup LDS.
//
// Layout: 8 octave slots x 8x8 corners each = 512 vec2<f32> entries (~4 KB).
// 256 threads x 2 loads = exactly 512 total loads, flushed with a single barrier.
//
// Coverage: for scale <= 0.0146 with lacunarity=2, all 6 octave corners fit in
// the 8x8 slot; beyond that the bounds check falls through to LUT-global
// (still no trig, just no LDS reuse for those corners).

// SquirrelNoise5 constants
const SQ5_BIT_NOISE1: u32 = 0xd2a80a3fu;
const SQ5_BIT_NOISE2: u32 = 0xa884f197u;
const SQ5_BIT_NOISE3: u32 = 0x6C736F4Bu;
const SQ5_BIT_NOISE4: u32 = 0xB79F3ABBu;
const SQ5_BIT_NOISE5: u32 = 0x1b56c4f5u;
const SQ5_PRIME1: i32 = 198491317;

fn squirrel5(n: i32, seed: u32) -> u32 {
    var mangled = u32(n);
    mangled = mangled * SQ5_BIT_NOISE1;
    mangled = mangled + seed;
    mangled ^= (mangled >> 9u);
    mangled = mangled + SQ5_BIT_NOISE2;
    mangled ^= (mangled >> 11u);
    mangled = mangled * SQ5_BIT_NOISE3;
    mangled ^= (mangled >> 13u);
    mangled = mangled + SQ5_BIT_NOISE4;
    mangled ^= (mangled >> 15u);
    mangled = mangled * SQ5_BIT_NOISE5;
    mangled ^= (mangled >> 17u);
    return mangled;
}

fn squirrel5_2d(x: i32, y: i32, seed: u32) -> u32 {
    return squirrel5(x + (y * SQ5_PRIME1), seed);
}

fn fade(t: f32) -> f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn fade_derivative(t: f32) -> f32 {
    return 30.0 * t * t * (t * (t - 2.0) + 1.0);
}

fn lerp_v(a: f32, b: f32, t: f32) -> f32 {
    return a + t * (b - a);
}

fn get_gradient(x: i32, y: i32, seed: u32) -> vec2<f32> {
    let hash = squirrel5_2d(x, y, seed);
    return gradient_lut[hash & 255u];
}

struct ComputeUniforms {
    width: u32,
    height: u32,
    scale: f32,
    seed: u32,
    octaves: u32,
    persistence: f32,
    lacunarity: f32,
    _padding: u32,
};

@group(0) @binding(0) var<uniform> uniforms: ComputeUniforms;
@group(0) @binding(1) var<storage, read_write> output_buffer: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> gradient_lut: array<vec2<f32>>;

// Shared gradient cache: MAX_OCTAVES octave slots, each holding an 8x8 grid of corners.
// Indexed as: shared_grads[oct * SLICE + ly * CORNERS_PER_DIM + lx]
const CORNERS_PER_DIM: u32 = 8u;
const SLICE: u32 = 64u;       // CORNERS_PER_DIM^2
const MAX_OCTAVES: u32 = 8u;  // must be >= uniforms.octaves

var<workgroup> shared_grads: array<vec2<f32>, 512u>; // MAX_OCTAVES * SLICE

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid_v: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let lid = lid_v.y * 16u + lid_v.x; // 0..255

    // ---- Cooperative load phase ----
    // Each thread loads exactly 2 slots (256 x 2 = 512 = MAX_OCTAVES x SLICE).
    // Slot n maps to octave oct = n/SLICE, local corner (lx, ly) = (within%8, within/8).
    // freq for octave oct = lacunarity^oct, recomputed via a short loop (<=7 muls).
    // Threads within a 32-wide warp always share the same oct, so no warp divergence.
    for (var p = 0u; p < 2u; p = p + 1u) {
        let n = lid + p * 256u;
        let oct = n / SLICE;
        let within = n % SLICE;
        let lx = i32(within % CORNERS_PER_DIM);
        let ly = i32(within / CORNERS_PER_DIM);

        var freq = 1.0;
        for (var k = 0u; k < oct; k = k + 1u) {
            freq = freq * uniforms.lacunarity;
        }

        let bx = i32(floor(f32(wgid.x * 16u) * uniforms.scale * freq));
        let by = i32(floor(f32(wgid.y * 16u) * uniforms.scale * freq));
        shared_grads[n] = get_gradient(bx + lx, by + ly, uniforms.seed + oct);
    }

    workgroupBarrier();

    if gid.x >= uniforms.width || gid.y >= uniforms.height { return; }

    let world_pos = vec2<f32>(f32(gid.x), f32(gid.y)) * uniforms.scale;

    var height = 0.0;
    var grad = vec2<f32>(0.0);
    var amplitude = 1.0;
    var freq = 1.0;

    for (var oct = 0u; oct < uniforms.octaves; oct = oct + 1u) {
        let pos = world_pos * freq;
        let iv = floor(pos);
        let fv = pos - iv;
        let cx = i32(iv.x);
        let cy = i32(iv.y);

        // Base corner of the cached 8x8 block for this octave
        let bx = i32(floor(f32(wgid.x * 16u) * uniforms.scale * freq));
        let by = i32(floor(f32(wgid.y * 16u) * uniforms.scale * freq));
        let lx0 = cx - bx;
        let ly0 = cy - by;

        var g00: vec2<f32>;
        var g10: vec2<f32>;
        var g01: vec2<f32>;
        var g11: vec2<f32>;

        // lx0 < CORNERS_PER_DIM-1 ensures cx+1 is also within the cached block
        if lx0 >= 0 && lx0 < i32(CORNERS_PER_DIM) - 1 &&
           ly0 >= 0 && ly0 < i32(CORNERS_PER_DIM) - 1 {
            let base_idx = oct * SLICE + u32(ly0) * CORNERS_PER_DIM + u32(lx0);
            g00 = shared_grads[base_idx];
            g10 = shared_grads[base_idx + 1u];
            g01 = shared_grads[base_idx + CORNERS_PER_DIM];
            g11 = shared_grads[base_idx + CORNERS_PER_DIM + 1u];
        } else {
            // Fallback: LUT-global (still no trig; only triggered outside default params)
            g00 = get_gradient(cx, cy, uniforms.seed + oct);
            g10 = get_gradient(cx + 1, cy, uniforms.seed + oct);
            g01 = get_gradient(cx, cy + 1, uniforms.seed + oct);
            g11 = get_gradient(cx + 1, cy + 1, uniforms.seed + oct);
        }

        let u = fade(fv.x);
        let v = fade(fv.y);
        let du = fade_derivative(fv.x);
        let dv = fade_derivative(fv.y);

        let a = dot(g00, fv);
        let b = dot(g10, fv - vec2<f32>(1.0, 0.0));
        let c = dot(g01, fv - vec2<f32>(0.0, 1.0));
        let d = dot(g11, fv - vec2<f32>(1.0, 1.0));

        let val = a + u * (b - a) + v * (c - a) + u * v * (a - b - c + d);
        let g_perlin = g00 + u * (g10 - g00) + v * (g01 - g00) + u * v * (g00 - g10 - g01 + g11)
                     + vec2<f32>(du * lerp_v(b - a, d - c, v), dv * lerp_v(c - a, d - b, u));

        height = height + val * amplitude;
        grad = grad + g_perlin * (amplitude * freq);

        amplitude = amplitude * uniforms.persistence;
        freq = freq * uniforms.lacunarity;
    }

    output_buffer[gid.y * uniforms.width + gid.x] = vec4<f32>(height, grad.x, grad.y, 0.0);
}
