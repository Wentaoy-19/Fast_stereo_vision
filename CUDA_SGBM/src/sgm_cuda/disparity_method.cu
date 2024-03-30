/**
    This file is part of sgm. (https://github.com/dhernandez0/sgm).

    Copyright (c) 2016 Daniel Hernandez Juarez.

    sgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    sgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with sgm.  If not, see <http://www.gnu.org/licenses/>.

**/

#include "disparity_method.h"
#include "torch/types.h"
#include <cstdint>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor sgm_cuda_forward(torch::Tensor left, torch::Tensor right, uint8_t p1, uint8_t p2)
{
	uint32_t cols, rows, size, size_cube_l;
	cols = left.size(1);
	rows = left.size(0);
	size = rows*cols;
	size_cube_l = size*MAX_DISPARITY;

	torch::Tensor transform0 = torch::zeros({rows, cols}, torch::dtype(torch::kInt32).device(left.device()));
	torch::Tensor transform1 = torch::zeros({rows, cols}, torch::dtype(torch::kInt32).device(left.device()));
	torch::Tensor cost = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kUInt8).device(left.device()));

	torch::Tensor disparity = torch::zeros({rows, cols}, torch::dtype(torch::kUInt8).device(left.device()));
	torch::Tensor disparity_filtered_uchar = torch::zeros({rows, cols}, torch::dtype(torch::kUInt8).device(left.device()));
	torch::Tensor S = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kInt16).device(left.device()));
	torch::Tensor L0 = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kUInt8).device(left.device()));
	torch::Tensor L1 = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kUInt8).device(left.device()));
	torch::Tensor L2 = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kUInt8).device(left.device()));
	torch::Tensor L3 = torch::zeros({rows, cols, MAX_DISPARITY}, torch::dtype(torch::kUInt8).device(left.device()));

	uint8_t *d_im0 = (uint8_t *)left.data_ptr();
	uint8_t *d_im1 = (uint8_t *)right.data_ptr();
	cost_t *d_transform0 = (cost_t *)transform0.data_ptr();
	cost_t *d_transform1 = (cost_t *)transform1.data_ptr();
	uint8_t *d_cost = (uint8_t *)cost.data_ptr();
	uint8_t *d_disparity = (uint8_t *)disparity.data_ptr();
	uint8_t *d_disparity_filtered_uchar = (uint8_t *)disparity_filtered_uchar.data_ptr();
	uint16_t *d_S = (uint16_t *)S.data_ptr();
	uint8_t *d_L0 = (uint8_t *)L0.data_ptr();
	uint8_t *d_L1 = (uint8_t *)L1.data_ptr();
	uint8_t *d_L2 = (uint8_t *)L2.data_ptr();
	uint8_t *d_L3 = (uint8_t *)L3.data_ptr();
	uint8_t	*d_L4 = nullptr;
	uint8_t	*d_L5 = nullptr;
	uint8_t	*d_L6 = nullptr;
	uint8_t	*d_L7 = nullptr;

    const at::cuda::OptionalCUDAGuard guard(device_of(left));

	dim3 block_size;
	block_size.x = 32;
	block_size.y = 32;

	dim3 grid_size;
	grid_size.x = (cols+block_size.x-1) / block_size.x;
	grid_size.y = (rows+block_size.y-1) / block_size.y;

	CenterSymmetricCensusKernelSM2<<<grid_size, block_size, 0>>>(d_im0, d_im1, d_transform0, d_transform1, rows, cols);

	HammingDistanceCostKernel<<<rows, MAX_DISPARITY, 0>>>(d_transform0, d_transform1, d_cost, rows, cols);

	const int PIXELS_PER_BLOCK = COSTAGG_BLOCKSIZE/WARP_SIZE;
	const int PIXELS_PER_BLOCK_HORIZ = COSTAGG_BLOCKSIZE_HORIZ/WARP_SIZE;

	CostAggregationKernelLeftToRight<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0>>>(d_cost, d_L0, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

	CostAggregationKernelRightToLeft<<<(rows+PIXELS_PER_BLOCK_HORIZ-1)/PIXELS_PER_BLOCK_HORIZ, COSTAGG_BLOCKSIZE_HORIZ, 0>>>(d_cost, d_L1, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

	CostAggregationKernelUpToDown<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0>>>(d_cost, d_L2, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

	CostAggregationKernelDownToUp<<<(cols+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK, COSTAGG_BLOCKSIZE, 0>>>(d_cost, d_L3, p1, p2, rows, cols, d_transform0, d_transform1, d_disparity, d_L0, d_L1, d_L2, d_L3, d_L4, d_L5, d_L6);

	MedianFilter3x3<<<(size+MAX_DISPARITY-1)/MAX_DISPARITY, MAX_DISPARITY, 0>>>(d_disparity, d_disparity_filtered_uchar, rows, cols);

	return disparity_filtered_uchar;
}
