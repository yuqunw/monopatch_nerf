#ifndef _QFF_ENCODE_H
#define _QFF_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>
#include <torch/extension.h>

void qff_forward(
    const at::Tensor points,       // Bx3
    const at::Tensor freqs,        // F
    const at::Tensor features,     // Fx8xCxRxRxR
    at::Tensor outputs,            // BxFx8xC
    at::Tensor dy_dx_cache,        // BxFx8xC
    const uint32_t B, 
    const uint32_t F, 
    const uint32_t C, 
    const uint32_t R);

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
    const uint32_t R);

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
    const uint32_t R);
#endif