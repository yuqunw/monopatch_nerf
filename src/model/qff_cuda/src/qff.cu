#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_ALL(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// requires CUDA >= 10 and ARCH >= 70
// this is very slow compared to float or __half2, do not use!
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}

template <typename T>
__device__ T trilinear_interp(
    const T * __restrict__ features, 
    const uint32_t R, 
    const T sx, 
    const T sy, 
    const T sz){
    const T xc = ((sx + 1) * 0.5) * (R - 1);
    const T yc = ((sy + 1) * 0.5) * (R - 1);
    const T zc = ((sz + 1) * 0.5) * (R - 1);

    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx = xc - (T) x0;
    const T wy = yc - (T) y0;
    const T wz = zc - (T) z0;

    // fxyz
    const T f000 = features[z0 * R * R + y0 * R + x0];
    const T f001 = features[z1 * R * R + y0 * R + x0];
    const T f010 = features[z0 * R * R + y1 * R + x0];
    const T f011 = features[z1 * R * R + y1 * R + x0];
    const T f100 = features[z0 * R * R + y0 * R + x1];
    const T f101 = features[z1 * R * R + y0 * R + x1];
    const T f110 = features[z0 * R * R + y1 * R + x1];
    const T f111 = features[z1 * R * R + y1 * R + x1];

    const T f00 = f000 * (1 - wx) + f100 * wx;
    const T f01 = f001 * (1 - wx) + f101 * wx;
    const T f10 = f010 * (1 - wx) + f110 * wx;
    const T f11 = f011 * (1 - wx) + f111 * wx;

    const T f0 = f00 * (1 - wy) + f10 * wy;
    const T f1 = f01 * (1 - wy) + f11 * wy;

    const T f = f0 * (1 - wz) + f1 * wz;

    return f;
}
template <typename T>
__device__ void grad_trilinear_interp(
    T * __restrict__ grad_features, 
    const uint32_t R, 
    const T sx, 
    const T sy, 
    const T sz, 
    const T grad_output
){

    const T xc = ((sx + 1) * 0.5) * (R - 1);
    const T yc = ((sy + 1) * 0.5) * (R - 1);
    const T zc = ((sz + 1) * 0.5) * (R - 1);
    
    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx = xc - (T)x0;
    const T wy = yc - (T)y0;
    const T wz = zc - (T)z0;

    // apply gradient
    atomicAdd(grad_features + z0 * R * R + y0 * R + x0, grad_output * (1 - wx) * (1 - wy) * (1 - wz));
    atomicAdd(grad_features + z1 * R * R + y0 * R + x0, grad_output * (1 - wx) * (1 - wy) * (0 + wz));
    atomicAdd(grad_features + z0 * R * R + y1 * R + x0, grad_output * (1 - wx) * (0 + wy) * (1 - wz));
    atomicAdd(grad_features + z1 * R * R + y1 * R + x0, grad_output * (1 - wx) * (0 + wy) * (0 + wz));
    atomicAdd(grad_features + z0 * R * R + y0 * R + x1, grad_output * (0 + wx) * (1 - wy) * (1 - wz));
    atomicAdd(grad_features + z1 * R * R + y0 * R + x1, grad_output * (0 + wx) * (1 - wy) * (0 + wz));
    atomicAdd(grad_features + z0 * R * R + y1 * R + x1, grad_output * (0 + wx) * (0 + wy) * (1 - wz));
    atomicAdd(grad_features + z1 * R * R + y1 * R + x1, grad_output * (0 + wx) * (0 + wy) * (0 + wz));
}

template <typename T>
__device__ void grad_point_helper(
    T * __restrict__ grad_points, 
    const T * __restrict__ features, 
    const uint32_t R, 
    const T sx, 
    const T sy, 
    const T sz, 
    const T dsx, 
    const T dsy, 
    const T dsz, 
    const T grad_output
){
    const T xc = ((sx + 1) * 0.5) * (R - 1);
    const T yc = ((sy + 1) * 0.5) * (R - 1);
    const T zc = ((sz + 1) * 0.5) * (R - 1);

    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx1 = xc - (T) x0;
    const T wy1 = yc - (T) y0;
    const T wz1 = zc - (T) z0;
    const T wx0 = 1 - wx1;
    const T wy0 = 1 - wy1;
    const T wz0 = 1 - wz1;

    // fxyz
    const T f000 = features[z0 * R * R + y0 * R + x0];
    const T f001 = features[z1 * R * R + y0 * R + x0];
    const T f010 = features[z0 * R * R + y1 * R + x0];
    const T f011 = features[z1 * R * R + y1 * R + x0];
    const T f100 = features[z0 * R * R + y0 * R + x1];
    const T f101 = features[z1 * R * R + y0 * R + x1];
    const T f110 = features[z0 * R * R + y1 * R + x1];
    const T f111 = features[z1 * R * R + y1 * R + x1];


    const T dwx0 = -dsx * 0.5 * (R - 1);
    const T dwx1 = dsx * 0.5 * (R - 1);
    const T dwy0 = -dsy * 0.5 * (R - 1);
    const T dwy1 = dsy * 0.5 * (R - 1);
    const T dwz0 = -dsz * 0.5 * (R - 1);
    const T dwz1 = dsz * 0.5 * (R - 1);


    const T df_dx = dwx0 * (f000 * wy0 * wz0 + f001 * wy0 * wz1 + f010 * wy1 * wz0 + f011 * wy1 * wz1) + \
                    dwx1 * (f100 * wy0 * wz0 + f101 * wy0 * wz1 + f110 * wy1 * wz0 + f111 * wy1 * wz1);

    const T df_dy = dwy0 * (f000 * wx0 * wz0 + f001 * wx0 * wz1 + f100 * wx1 * wz0 + f101 * wx1 * wz1) + \
                    dwy1 * (f010 * wx0 * wz0 + f011 * wx0 * wz1 + f110 * wx1 * wz0 + f111 * wx1 * wz1);

    const T df_dz = dwz0 * (f000 * wx0 * wy0 + f010 * wx0 * wy1 + f100 * wx1 * wy0 + f110 * wx1 * wy1) + \
                    dwz1 * (f001 * wx0 * wy0 + f011 * wx0 * wy1 + f101 * wx1 * wy0 + f111 * wx1 * wy1);

    atomicAdd(grad_points + 0, grad_output * df_dx);
    atomicAdd(grad_points + 1, grad_output * df_dy);
    atomicAdd(grad_points + 2, grad_output * df_dz);
}

template <typename T>
__device__ void grad_grad_helper(
    const T * __restrict__ features, 
    T * __restrict__ grad2_features, 
    T * __restrict__ grad_grad_output, 
    const uint32_t R, 
    const T sx, 
    const T sy, 
    const T sz, 
    const T dsx, 
    const T dsy, 
    const T dsz, 
    const T gx, 
    const T gy, 
    const T gz, 
    const T grad_output
){

    const T xc = ((sx + 1) * 0.5) * (R - 1);
    const T yc = ((sy + 1) * 0.5) * (R - 1);
    const T zc = ((sz + 1) * 0.5) * (R - 1);
    
    const uint32_t x0 = min(max((uint32_t)floor(xc), 0), R-1);
    const uint32_t y0 = min(max((uint32_t)floor(yc), 0), R-1);
    const uint32_t z0 = min(max((uint32_t)floor(zc), 0), R-1);
    const uint32_t x1 = max(min((uint32_t)ceil(xc), R-1), 0);
    const uint32_t y1 = max(min((uint32_t)ceil(yc), R-1), 0);
    const uint32_t z1 = max(min((uint32_t)ceil(zc), R-1), 0);

    const T wx = xc - (T)x0;
    const T wy = yc - (T)y0;
    const T wz = zc - (T)z0;

    const T wx1 = wx;
    const T wy1 = wy;
    const T wz1 = wz;

    const T wx0 = (1 - wx);
    const T wy0 = (1 - wy);
    const T wz0 = (1 - wz);

    const T dwx0 = -dsx * 0.5 * (R - 1) * gx;
    const T dwx1 = dsx * 0.5 * (R - 1) * gx;
    const T dwy0 = -dsy * 0.5 * (R - 1) * gy;
    const T dwy1 = dsy * 0.5 * (R - 1) * gy;
    const T dwz0 = -dsz * 0.5 * (R - 1) * gz;
    const T dwz1 = dsz * 0.5 * (R - 1) * gz;

    // apply gradient
    //                                                                  x             y             z
    atomicAdd(grad2_features + z0 * R * R + y0 * R + x0, grad_output * ( dwx0*wy0*wz0 +dwy0*wx0*wz0 +dwz0*wx0*wy0));
    atomicAdd(grad2_features + z1 * R * R + y0 * R + x0, grad_output * ( dwx0*wy0*wz1 +dwy0*wx0*wz1 +dwz1*wx0*wy0));
    atomicAdd(grad2_features + z0 * R * R + y1 * R + x0, grad_output * ( dwx0*wy1*wz0 +dwy1*wx0*wz0 +dwz0*wx0*wy1));
    atomicAdd(grad2_features + z1 * R * R + y1 * R + x0, grad_output * ( dwx0*wy1*wz1 +dwy1*wx0*wz1 +dwz1*wx0*wy1));
    atomicAdd(grad2_features + z0 * R * R + y0 * R + x1, grad_output * ( dwx1*wy0*wz0 +dwy0*wx1*wz0 +dwz0*wx1*wy0));
    atomicAdd(grad2_features + z1 * R * R + y0 * R + x1, grad_output * ( dwx1*wy0*wz1 +dwy0*wx1*wz1 +dwz1*wx1*wy0));
    atomicAdd(grad2_features + z0 * R * R + y1 * R + x1, grad_output * ( dwx1*wy1*wz0 +dwy1*wx1*wz0 +dwz0*wx1*wy1));
    atomicAdd(grad2_features + z1 * R * R + y1 * R + x1, grad_output * ( dwx1*wy1*wz1 +dwy1*wx1*wz1 +dwz1*wx1*wy1));


    // fxyz
    const T f000 = features[z0 * R * R + y0 * R + x0];
    const T f001 = features[z1 * R * R + y0 * R + x0];
    const T f010 = features[z0 * R * R + y1 * R + x0];
    const T f011 = features[z1 * R * R + y1 * R + x0];
    const T f100 = features[z0 * R * R + y0 * R + x1];
    const T f101 = features[z1 * R * R + y0 * R + x1];
    const T f110 = features[z0 * R * R + y1 * R + x1];
    const T f111 = features[z1 * R * R + y1 * R + x1];


    const T df_dx = dwx0 * (f000 * wy0 * wz0 + f001 * wy0 * wz1 + f010 * wy1 * wz0 + f011 * wy1 * wz1) + \
                    dwx1 * (f100 * wy0 * wz0 + f101 * wy0 * wz1 + f110 * wy1 * wz0 + f111 * wy1 * wz1);

    const T df_dy = dwy0 * (f000 * wx0 * wz0 + f001 * wx0 * wz1 + f100 * wx1 * wz0 + f101 * wx1 * wz1) + \
                    dwy1 * (f010 * wx0 * wz0 + f011 * wx0 * wz1 + f110 * wx1 * wz0 + f111 * wx1 * wz1);

    const T df_dz = dwz0 * (f000 * wx0 * wy0 + f010 * wx0 * wy1 + f100 * wx1 * wy0 + f110 * wx1 * wy1) + \
                    dwz1 * (f001 * wx0 * wy0 + f011 * wx0 * wy1 + f101 * wx1 * wy0 + f111 * wx1 * wy1);

    atomicAdd(grad_grad_output, df_dx + df_dy + df_dz);
}

template <typename T>
__global__ void kernel_qff_forward(
    const T * __restrict__ points,       // Bx3
    const T * __restrict__ freqs,        // F
    const T * __restrict__ features,     // Fx8xCxRxRxR
    T * __restrict__ outputs,            // BxFx8xC
    T * __restrict__ dy_dx_cache,        // BxFx8xC
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R) {
    const uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
	if (bf>= B*F) return;
    const uint32_t b = bf / F;
    const uint32_t f = bf % F;
    const uint32_t RRR = R*R*R;

    points += b*3;
    features += f*2*C*RRR;
    outputs += b * F * 2 * C + f * 2 * C;


    const T freq = freqs[f];

    // first compute sinusoidal coeffs
    const T px = points[0];
    const T py = points[1];
    const T pz = points[2];

    const T sx = sin(freq * px);
    const T sy = sin(freq * py);
    const T sz = sin(freq * pz);
    const T cx = cos(freq * px);
    const T cy = cos(freq * py);
    const T cz = cos(freq * pz);

    for (uint32_t c = 0; c < C; c++){
        const T* f = features + c * RRR;
        outputs[c + 0 * C]= trilinear_interp(f + 0 * C * RRR, R, sx, sy, sz);
        outputs[c + 1 * C]= trilinear_interp(f + 1 * C * RRR, R, cx, cy, cz);
    }
}


template <typename T>
__global__ void kernel_qff_backward(
    const T * __restrict__ grad_output,  // BxFx8xC
    const T * __restrict__ points,       // Bx3
    const T * __restrict__ freqs,        // F
    const T * __restrict__ features,     // Fx8xCxRxRxR
    T * __restrict__ grad_points,        // Bx3
    T * __restrict__ grad_features,      // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R
) {
    const uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
	if (bf>= B*F) return;
    const uint32_t b = bf / F;
    const uint32_t f = bf % F;
    const uint32_t RRR = R*R*R;

    points += b*3;
    features += f*2*C*R*R*R;

    // setup gradient offset
    grad_points += b*3;
    grad_features += f*2*C*R*R*R;
    grad_output += b * F * 2 * C + f * 2 * C;

    const T freq = freqs[f];

    // first compute sinusoidal coeffs
    const T px = points[0];
    const T py = points[1];
    const T pz = points[2];

    const T sx = sin(freq * px);
    const T sy = sin(freq * py);
    const T sz = sin(freq * pz);
    const T cx = cos(freq * px);
    const T cy = cos(freq * py);
    const T cz = cos(freq * pz);

    const T dsx = freq * cos(freq * px);
    const T dsy = freq * cos(freq * py);
    const T dsz = freq * cos(freq * pz);
    const T dcx = -freq * sin(freq * px);
    const T dcy = -freq * sin(freq * py);
    const T dcz = -freq * sin(freq * pz);

    for (uint32_t c = 0; c < C; c++){
        T* gf = grad_features + c * R*R*R;
        const T* fe = features + c * R*R*R;

        // compute grad features
        grad_trilinear_interp(gf + 0 * C * RRR, R, sx, sy, sz, grad_output[c + 0 * C]);
        grad_trilinear_interp(gf + 1 * C * RRR, R, cx, cy, cz, grad_output[c + 1 * C]);

        // compute grad points
        // f = f
        grad_point_helper(grad_points, fe + 0 * C * RRR, R, sx, sy, sz, dsx, dsy, dsz, grad_output[c + 0 * C]);
        grad_point_helper(grad_points, fe + 1 * C * RRR, R, cx, cy, cz, dcx, dcy, dcz, grad_output[c + 1 * C]);
    }
}
template <typename T>
__global__ void kernel_qff_backward_backward(
    const T * __restrict__ grad_output,       // BxFx8xC
    const T * __restrict__ grad_grad_points,  // Bx3
    const T * __restrict__ points,            // Bx3
    const T * __restrict__ freqs,             // F
    const T * __restrict__ features,          // Fx8xCxRxRxR
    T * __restrict__ grad_grad_outputs,       // BxFx8xC
    T * __restrict__ grad2_features,          // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R) {
    const uint32_t bf = blockIdx.x * blockDim.x + threadIdx.x;
	if (bf>= B*F) return;

    const uint32_t b = bf / F;
    const uint32_t f = bf % F;
    const uint32_t RRR = R*R*R;

    points += b*3;
    features += f*2*C*R*R*R;

    // setup gradient offset
    grad_grad_points += b*3;
    grad2_features += f*2*C*R*R*R;
    grad_output += b * F * 2 * C + f * 2 * C;
    grad_grad_outputs += b * F * 2 * C + f * 2 * C;

    const T freq = freqs[f];

    // first compute sinusoidal coeffs
    const T px = points[0];
    const T py = points[1];
    const T pz = points[2];

    const T sx = sin(freq * px);
    const T sy = sin(freq * py);
    const T sz = sin(freq * pz);
    const T cx = cos(freq * px);
    const T cy = cos(freq * py);
    const T cz = cos(freq * pz);

    const T dsx = freq * cos(freq * px);
    const T dsy = freq * cos(freq * py);
    const T dsz = freq * cos(freq * pz);
    const T dcx = -freq * sin(freq * px);
    const T dcy = -freq * sin(freq * py);
    const T dcz = -freq * sin(freq * pz);

    const T gpx = grad_grad_points[0];
    const T gpy = grad_grad_points[1];
    const T gpz = grad_grad_points[2];


    for (uint32_t c = 0; c < C; c++){
        T* gf = grad2_features + c * R*R*R;
        const T* f = features + c * R*R*R;

        grad_grad_helper(f + 0 * C * RRR, gf + 0 * C * RRR, grad_grad_outputs + c + 0 * C , R, sx, sy, sz, dsx, dsy, dsz, gpx, gpy, gpz, grad_output[c + 0 * C]);
        grad_grad_helper(f + 1 * C * RRR, gf + 1 * C * RRR, grad_grad_outputs + c + 1 * C , R, cx, cy, cz, dcx, dcy, dcz, gpx, gpy, gpz, grad_output[c + 1 * C]);
    }
}


///////////////////////////////
// CUDA device call wrappers

template <typename T>
void qff_forward_cuda(
    const T * points,       // Bx3
    const T * freqs,        // F
    const T * features,     // Fx8xCxRxRxR
    T * outputs,            // BxFx8xC
    T * dy_dx_cache,        // BxFx8xC
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R) {

    static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks = { div_round_up(B*F, N_THREAD), 1, 1 };
    kernel_qff_forward<T><<<blocks, N_THREAD>>>( points, freqs, features, outputs, dy_dx_cache, B, F, C, R); 
}
template <typename T>
void qff_backward_cuda(
    const T * grad_output,  // BxFx8xC
    const T * points,       // Bx3
    const T * freqs,        // F
    const T * features,     // Fx8xCxRxRxR
    T * grad_points,        // Bx3
    T * grad_features,      // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R) {

    static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks = { div_round_up(B*F, N_THREAD), 1, 1 };
    kernel_qff_backward<T><<<blocks, N_THREAD>>>(grad_output, points, freqs, features, grad_points, grad_features, B, F, C, R); 
}

template <typename T>
void qff_backward_backward_cuda(
    const T * grad_output,       // BxFx8xC
    const T * grad_grad_points,  // Bx3
    const T * points,            // Bx3
    const T * freqs,             // F
    const T * features,          // Fx8xCxRxRxR
    T * grad_grad_outputs,       // BxFx8xC
    T * grad2_features,          // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R) {
    static constexpr uint32_t N_THREAD = 256;
	const dim3 blocks = { div_round_up(B*F, N_THREAD), 1, 1 };
    kernel_qff_backward_backward<T><<<blocks, N_THREAD>>>(grad_output, grad_grad_points, points, freqs, features, grad_grad_outputs, grad2_features, B, F, C, R); 
}

///////////////////////////////
// CUDA template call wrappers

void qff_forward(
    const at::Tensor points,       // Bx3
    const at::Tensor freqs,        // F
    const at::Tensor features,     // Fx8xCxRxRxR
    at::Tensor outputs,            // BxFx8xC
    at::Tensor dy_dx_cache,        // BxFx8xC
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R
){

    CHECK_ALL(points);
    CHECK_ALL(freqs);
    CHECK_ALL(features);
    CHECK_ALL(outputs);
    CHECK_ALL(dy_dx_cache);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        points.scalar_type(), "qff_forward", (
            [&] {
                qff_forward_cuda<scalar_t>(
                    points.data_ptr<scalar_t>(), 
                    freqs.data_ptr<scalar_t>(), 
                    features.data_ptr<scalar_t>(), 
                    outputs.data_ptr<scalar_t>(), 
                    dy_dx_cache.data_ptr<scalar_t>(), 
                    B, F, C, R
                );
            }
        )
    );
}
void qff_backward(
    const at::Tensor grad_output,  // BxFx8xC
    const at::Tensor points,       // Bx3
    const at::Tensor freqs,        // F
    const at::Tensor features,     // Fx8xCxRxRxR
    at::Tensor grad_points,        // Bx3
    at::Tensor grad_features,      // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R
){

    CHECK_ALL(grad_output);
    CHECK_ALL(points);
    CHECK_ALL(freqs);
    CHECK_ALL(features);
    CHECK_ALL(grad_points);
    CHECK_ALL(grad_features);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        points.scalar_type(), "qff_backward", (
            [&] {
                qff_backward_cuda<scalar_t>(
                    grad_output.data_ptr<scalar_t>(), 
                    points.data_ptr<scalar_t>(), 
                    freqs.data_ptr<scalar_t>(), 
                    features.data_ptr<scalar_t>(), 
                    grad_points.data_ptr<scalar_t>(), 
                    grad_features.data_ptr<scalar_t>(), 
                    B, F, C, R
                );
            }
        )
    );
}

// _backend.qff_backward_backward(grad_outputs, grad_grad_points, points, freqs, features, grad_grad_outputs, grad2_features, B, F, C, R)
void qff_backward_backward(
    const at::Tensor grad_output,       // BxFx8xC
    const at::Tensor grad_grad_points,  // Bx3
    const at::Tensor points,            // Bx3
    const at::Tensor freqs,             // F
    const at::Tensor features,          // Fx8xCxRxRxR
    at::Tensor grad_grad_outputs,       // BxFx8xC
    at::Tensor grad2_features,          // Fx8xCxRxRxR
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R
){

    CHECK_ALL(grad_output);
    CHECK_ALL(grad_grad_points);
    CHECK_ALL(points);
    CHECK_ALL(freqs);
    CHECK_ALL(features);
    CHECK_ALL(grad_grad_outputs);
    CHECK_ALL(grad2_features);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        points.scalar_type(), "qff_backward_backward", (
            [&] {
                qff_backward_backward_cuda<scalar_t>(
                    grad_output.data_ptr<scalar_t>(), 
                    grad_grad_points.data_ptr<scalar_t>(), 
                    points.data_ptr<scalar_t>(), 
                    freqs.data_ptr<scalar_t>(), 
                    features.data_ptr<scalar_t>(), 
                    grad_grad_outputs.data_ptr<scalar_t>(), 
                    grad2_features.data_ptr<scalar_t>(), 
                    B, F, C, R
                );
            }
        )
    );
}