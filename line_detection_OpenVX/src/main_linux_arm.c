/*
 *
 * Copyright (c) 2017 Texas Instruments Incorporated
 *
 * All rights reserved not granted herein.
 *
 * Limited License.
 *
 * Texas Instruments Incorporated grants a world-wide, royalty-free, non-exclusive
 * license under copyrights and patents it now or hereafter owns or controls to make,
 * have made, use, import, offer to sell and sell ("Utilize") this software subject to the
 * terms herein.  With respect to the foregoing patent license, such license is granted
 * solely to the extent that any such patent is necessary to Utilize the software alone.
 * The patent license shall not apply to any combinations which include this software,
 * other than combinations with devices manufactured by or for TI ("TI Devices").
 * No hardware patent is licensed hereunder.
 *
 * Redistributions must preserve existing copyright notices and reproduce this license
 * (including the above copyright notice and the disclaimer and (if applicable) source
 * code license limitations below) in the documentation and/or other materials provided
 * with the distribution
 *
 * Redistribution and use in binary form, without modification, are permitted provided
 * that the following conditions are met:
 *
 * *       No reverse engineering, decompilation, or disassembly of this software is
 * permitted with respect to any software provided in binary form.
 *
 * *       any redistribution and use are licensed by TI for use only with TI Devices.
 *
 * *       Nothing shall obligate TI to provide you with source code for the software
 * licensed and provided to you in object code.
 *
 * If software source code is provided to you, modification and redistribution of the
 * source code are permitted provided that the following conditions are met:
 *
 * *       any redistribution and use of the source code, including any resulting derivative
 * works, are licensed by TI for use only with TI Devices.
 *
 * *       any redistribution and use of any object code compiled from the source code
 * and any resulting derivative works, are licensed by TI for use only with TI Devices.
 *
 * Neither the name of Texas Instruments Incorporated nor the names of its suppliers
 *
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * DISCLAIMER.
 *
 * THIS SOFTWARE IS PROVIDED BY TI AND TI'S LICENSORS "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL TI AND TI'S LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <TI/tivx.h>
#include <app_init.h>
#include <stdlib.h>
#include <time.h>

#include <tivx_utils_graph_perf.h>
#include <utils/perf_stats/include/app_perf_stats.h>
#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include <VX/vx_types.h>
#include <VX/vxu.h>
#include <VX/vx_nodes.h>
#include "src/readImage.h"
#include "src/writeImage.h"
#include <math.h>

#define NUM_THETAS (200)
#define NUM_THETAS_HALF (100)
#define NUM_THETAS_LF (200.0)
#define NUM_RHOS (200)
#define NUM_RHOS_HALF (100)
#define SCALING (8.0)

#define PI (3.14159265359)

#define ERROR_CHECK_STATUS(status)                                                              \
    {                                                                                           \
        vx_status status_ = (status);                                                           \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

#define ERROR_CHECK_OBJECT(obj)                                                                 \
    {                                                                                           \
        vx_status status_ = vxGetStatus((vx_reference)(obj));                                   \
        if (status_ != VX_SUCCESS)                                                              \
        {                                                                                       \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1);                                                                            \
        }                                                                                       \
    }

////////
// User kernel should have a unique enumerations and name for user kernel:
//   USER_LIBRARY_EXAMPLE      - library ID for user kernels in this example
//   USER_KERNEL_MEDIAN_BLUR   - enumeration for "app.userkernels.median_blur" kernel
//
// TODO:********
//   1. Define USER_LIBRARY_EXAMPLE
//   2. Define USER_KERNEL_MEDIAN_BLUR using VX_KERNEL_BASE() macro
enum user_library_e
{
    USER_LIBRARY_EXAMPLE = 1,
};
enum user_kernel_e
{
    USER_KERNEL_HOUGH_TRANSFORM = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE) + 0x001,
    USER_KERNEL_COLORING = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE) + 0x002,
};

vx_node userHoughTransformNode(vx_graph graph,
                               vx_image input,
                               vx_array rects)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_HOUGH_TRANSFORM);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);

    vx_array cos_lut = vxCreateArray(context, VX_TYPE_FLOAT64, NUM_THETAS);
    vx_array sin_lut = vxCreateArray(context, VX_TYPE_FLOAT64, NUM_THETAS);

    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)rects));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference)cos_lut));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 3, (vx_reference)sin_lut));

    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return node;
}
vx_status VX_CALLBACK hough_validator(vx_node node,
                                      const vx_reference parameters[], vx_uint32 num,
                                      vx_meta_format metas[])
{
    // parameter #0 -- input image of format VX_DF_IMAGE_U8
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if (format != VX_DF_IMAGE_U8)
    {
        return VX_ERROR_INVALID_FORMAT;
    }

    // parameter #1 -- output array should be of itemtype VX_TYPE_RECTANGLE
    format = VX_TYPE_COORDINATES2D;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_ARRAY_ITEMTYPE, &format, sizeof(format)));
    if (format != VX_TYPE_COORDINATES2D)
    {
        return VX_ERROR_INVALID_FORMAT;
    }
    return VX_SUCCESS;
}


vx_status VX_CALLBACK hough_host_side_function(vx_node node, const vx_reference *refs, vx_uint32 num)
{
    vx_image input = (vx_image)refs[0];
    vx_array output_arr = (vx_array)refs[1];
    vx_array cos_lut = (vx_array)refs[2];
    vx_array sin_lut = (vx_array)refs[3];

    vx_context context = vxGetContext((vx_reference)node);

    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height)));

    vx_rectangle_t rect = {.start_x = 0, .start_y = 0, .end_x = width, .end_y = height};
    vx_map_id map_input;
    vx_imagepatch_addressing_t addr_input;
    void *ptr_input;
    ERROR_CHECK_STATUS(vxMapImagePatch(input, &rect, 0, &map_input, &addr_input, &ptr_input, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    /******************************************
     * Making mask
     * ****************************************/

    vx_rectangle_t line0 = {.start_x = 0, .start_y = width / 5, .end_x = height - 20, .end_y = 0};
    vx_float64 k0 = ((vx_float64)line0.end_y - (vx_float64)line0.start_y) / ((vx_float64)line0.end_x - (vx_float64)line0.start_x);
    vx_float64 n0 = (vx_int32)line0.start_y - k0 * (vx_int32)line0.start_x;

    vx_rectangle_t line1 = {.start_x = 0, .start_y = 4 * width / 5, .end_x = height - 20, .end_y = width};
    vx_float64 k1 = ((vx_float64)line1.end_y - (vx_float64)line1.start_y) / ((vx_float64)line1.end_x - (vx_float64)line1.start_x);
    vx_float64 n1 = (vx_int32)line1.start_y - k1 * (vx_int32)line1.start_x;
    vx_int32 curr_yy0;
    vx_int32 curr_yy1;

    /* more efficient direct addressing by client.
     * for subsampled planes, scale will change.
     */
    vx_uint32 x, y, i, j;
    vx_bool switched = vx_false_e;
    for (y = 0; y < addr_input.dim_y; y += addr_input.step_y)
    {
        switched = vx_false_e;
        vx_uint8 state = 0;
        curr_yy0 = k0 * y + n0;
        curr_yy1 = k1 * y + n1;
        j = (addr_input.stride_y * y * addr_input.scale_y) / VX_SCALE_UNITY;
        for (x = 0; x < addr_input.dim_x; x += addr_input.step_x)
        {
            vx_uint8 *tmp = (vx_uint8 *)ptr_input;
            if (x == curr_yy0 || x == curr_yy1)
            {
                if(switched == vx_true_e)
                    switched = vx_false_e;
                else
                    switched = vx_true_e;
            }
            if (switched == vx_false_e)
            {
                i = j + (addr_input.stride_x * x * addr_input.scale_x) /
                            VX_SCALE_UNITY;
                tmp[i] = state;
            }
        }
    }

    // Mat mat(height, width, CV_8UC1, ptr_input, addr_input.stride_y);
    // imshow("edged",mat);

    /********************************************
     * Creating accumulator
     * ******************************************/
    void *base_ptr = NULL;
    vx_rectangle_t rect_transformed;
    vx_imagepatch_addressing_t addr;
    vx_map_id map_id;
    rect_transformed.start_x = rect_transformed.start_y = 0;
    rect_transformed.end_x = NUM_RHOS;
    rect_transformed.end_y = NUM_THETAS;
    vx_image accum = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_S16);

    ERROR_CHECK_STATUS(vxMapImagePatch(accum, &rect_transformed, 0, &map_id,
                                       &addr, &base_ptr,
                                       VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    // for (y = 0; y < addr.dim_y; y += addr.step_y)
    // {
    //     j = (addr.stride_y * y * addr.scale_y) / VX_SCALE_UNITY;
    //     for (x = 0; x < addr.dim_x; x += addr.step_x)
    //     {
    //         vx_uint8 *tmp = (vx_uint8 *)base_ptr;
    //         i = j + (addr.stride_x * x * addr.scale_x) /
    //                     VX_SCALE_UNITY;
    //         tmp[i] = 0;
    //         tmp[i + 1] = 0;
    //     }
    // }

     for (vx_uint32 i = 0; i < addr.dim_x * addr.dim_y; i++)
    {
        vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress1d(base_ptr, i, &addr);
        *ptr2 = 0;
    }
    /*****************************************************************************/
    vx_map_id cos_lut_map_id;
    void *cos_lut_base_ptr = NULL;
    vx_uint64 stride;
    ERROR_CHECK_STATUS(vxMapArrayRange(cos_lut, 0, NUM_THETAS, &cos_lut_map_id, &stride, &cos_lut_base_ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));
    vx_float64 *trig_cos = (vx_float64 *)cos_lut_base_ptr;

    vx_map_id sin_lut_map_id;
    void *sin_lut_base_ptr = NULL;
    ERROR_CHECK_STATUS(vxMapArrayRange(sin_lut, 0, NUM_THETAS, &sin_lut_map_id, &stride, &sin_lut_base_ptr, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));
    vx_float64 *trig_sin = (vx_float64 *)sin_lut_base_ptr;

    for (vx_uint32 i = 0; i < width; i++)
    {
        // From 10 because canny edge detector has some problem with upper edge
        for (vx_uint32 j = 10; j < height; j++)
        {
            vx_uint8 *ptr2 = (vx_uint8 *)vxFormatImagePatchAddress2d(ptr_input, i, j, &addr_input);
            // 1. calculate variables for non-zero pixels
            if (*ptr2 != 0)
            {
                // do calculation
                for (vx_uint32 k = 0; k < NUM_THETAS; k++)
                {

                    vx_uint32 rho = NUM_RHOS_HALF + (vx_float64)((vx_float64)i * trig_cos[k] - (vx_float64)j * trig_sin[k]) / SCALING;
                    vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress2d(base_ptr, rho, k, &addr);
                    *ptr2 = *ptr2 + 1;
                }
            }
        }
    }

    vx_int32 left_max = 0, right_max = 0;
    vx_coordinates2d_t left_coord, right_coord;
    vx_bool left = vx_false_e, right = vx_false_e;
    for (vx_uint32 i = 0; i < NUM_THETAS_HALF; i++)
    {
        for (vx_uint32 j = 0; j < NUM_RHOS; j++)
        {
            vx_int16 *pxl_lft = (vx_int16 *)vxFormatImagePatchAddress2d(base_ptr, j, i, &addr);
            vx_int16 *pxl_rght = (vx_int16 *)vxFormatImagePatchAddress2d(base_ptr, j, i + NUM_THETAS_HALF, &addr);
            vx_float64 theta1 = (vx_float64)i * PI / NUM_THETAS_LF;
            vx_float64 theta2 = ((vx_float64)i + NUM_THETAS_HALF) * PI / NUM_THETAS_LF;
            if (*pxl_lft > left_max && sin(theta1) < 0.8)
            {
                left_max = *pxl_lft;
                left_coord.x = i;
                left_coord.y = j;
                left = vx_true_e;
            }
            if (*pxl_rght > right_max && sin(theta2) < 0.8)
            {
                right_max = *pxl_rght;
                right_coord.x = i + NUM_THETAS_HALF;
                right_coord.y = j;
                right = vx_true_e;
            }
        }
    }
    if (left == vx_true_e && right == vx_true_e)
    {
        vxAddArrayItems(output_arr, 1, &left_coord, sizeof(vx_coordinates2d_t));
        vxAddArrayItems(output_arr, 1, &right_coord, sizeof(vx_coordinates2d_t));
    }

    ERROR_CHECK_STATUS(vxUnmapImagePatch(accum, map_id));
    ERROR_CHECK_STATUS(vxReleaseImage(&accum));

    ERROR_CHECK_STATUS(vxUnmapArrayRange(cos_lut, cos_lut_map_id));
    ERROR_CHECK_STATUS(vxUnmapArrayRange(sin_lut, sin_lut_map_id));

    ERROR_CHECK_STATUS(vxUnmapImagePatch(input, map_input));
    //ERROR_CHECK_STATUS(vxReleaseImage(&input));

    return VX_SUCCESS;
}

vx_node userColorNode(vx_graph graph,
                      vx_image input,
                      vx_array rects,
                      vx_image output,
                      vx_rectangle_t roi)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_COLORING);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);

    vx_array roi_arr = vxCreateArray(context, VX_TYPE_RECTANGLE, 1);
    ERROR_CHECK_STATUS(vxAddArrayItems(roi_arr, 1, &roi, sizeof(vx_rectangle_t)));
    ERROR_CHECK_OBJECT(roi_arr);

    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)rects));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference)output));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 3, (vx_reference)roi_arr));

    ERROR_CHECK_STATUS(vxReleaseArray(&roi_arr));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return node;
}
vx_status VX_CALLBACK colorNodeValidator(vx_node node,
                                         const vx_reference parameters[], vx_uint32 num,
                                         vx_meta_format metas[])
{
    // parameter #0 -- input image of format VX_DF_IMAGE_U8
    vx_df_image format = VX_DF_IMAGE_VIRT;
    vx_uint32 in_width, in_height;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_FORMAT, &format, sizeof(format)));
    if (format != VX_DF_IMAGE_RGB)
    {
        return VX_ERROR_INVALID_FORMAT;
    }
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_WIDTH, &in_width, sizeof(in_width)));
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[0], VX_IMAGE_HEIGHT, &in_height, sizeof(in_height)));
    // parameter #1 -- output array should be of itemtype VX_TYPE_COORDINATES2D

    format = VX_TYPE_COORDINATES2D;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &format, sizeof(format)));
    if (format != VX_TYPE_COORDINATES2D)
    {
        return VX_ERROR_INVALID_FORMAT;
    }
    // parameter #2 -- output image should be VX_DF_IMAGE_RGB
    format = VX_DF_IMAGE_RGB;
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_FORMAT, &format, sizeof(format)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_WIDTH, &in_width, sizeof(in_width)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[2], VX_IMAGE_ATTRIBUTE_HEIGHT, &in_height, sizeof(in_height)));

    // parameter #3 -- input array should contain VX_TYPE_RECTANGLE
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &format, sizeof(format)));
    if (format != VX_TYPE_RECTANGLE)
    {
        return VX_ERROR_INVALID_TYPE;
    }

    return VX_SUCCESS;
}

void move_diagonally(void *base_rgb, vx_imagepatch_addressing_t rgb_imgpatch, vx_uint32 width, vx_uint32 height, vx_int32 x0, vx_int32 y0, vx_int32 x1, vx_int32 y1)
{

    vx_float64 k = ((vx_float64)y1 - (vx_float64)y0) / ((vx_float64)x1 - (vx_float64)x0);
    vx_float64 n = (vx_int32)y0 - k * (vx_int32)x0;
    vx_int32 curr_xx = x0;
    vx_int32 curr_yy;

    vx_uint32 y, i, j;
    vx_uint8 red = 255;
    vx_uint8 green = 0;
    vx_uint8 blue = 0;
    for (y = 0; y < rgb_imgpatch.dim_y; y += rgb_imgpatch.step_y, curr_xx++)
    {
        j = (rgb_imgpatch.stride_y * y * rgb_imgpatch.scale_y) / VX_SCALE_UNITY;
        curr_yy = k * curr_xx + n;
        if (curr_xx < 0 || curr_xx >= height || curr_yy < 2 || curr_yy >= width)
            continue;
        vx_uint8 *tmp = (vx_uint8 *)base_rgb;
        i = j + (rgb_imgpatch.stride_x * curr_yy * rgb_imgpatch.scale_x) /
                    VX_SCALE_UNITY;
        for (vx_uint32 tickness = 0; tickness < 3; tickness++)
        {
            tmp[i + tickness * rgb_imgpatch.stride_x] = red;
            tmp[i + 1 + tickness * rgb_imgpatch.stride_x] = green;
            tmp[i + 2 + tickness * rgb_imgpatch.stride_x] = blue;
        }
    }
}

vx_status VX_CALLBACK coloring_host_side_function(vx_node node, const vx_reference *refs, vx_uint32 num)
{
    vx_image input_img = (vx_image)refs[0];
    vx_array coords_arr = (vx_array)refs[1];
    vx_image output_img = (vx_image)refs[2];
    vx_array roi_arr = (vx_array)refs[3];


    vx_uint32 width, height;
    ERROR_CHECK_STATUS(vxQueryImage(input_img, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(vx_uint32)));
    ERROR_CHECK_STATUS(vxQueryImage(input_img, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(vx_uint32)));

    vx_map_id roi_mapid;
    vx_size roi_stride = sizeof(vx_rectangle_t);
    vx_uint32 roi_width, roi_height;
    vx_rectangle_t *roi_rect = NULL;
    vxMapArrayRange(roi_arr, 0, 1, &roi_mapid, &roi_stride, (void **)&roi_rect, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0);

    roi_width = roi_rect->end_x - roi_rect->start_x;
    roi_height = roi_rect->end_y - roi_rect->start_y;
    vx_rectangle_t roied_roi_rectangle = {.start_x = 0, .start_y = 0, .end_x = roi_width, .end_y = roi_height};
    vx_image cropped_in_img = vxCreateImageFromROI(input_img, roi_rect);
    ERROR_CHECK_OBJECT(cropped_in_img);

    /**************************************************
     * Mapping image R,G,B components
     * ************************************************/
    vx_map_id cropped_map_id;
    vx_imagepatch_addressing_t cropped_imgpatch;
    void *cropped_ptr;
    ERROR_CHECK_STATUS(vxMapImagePatch(cropped_in_img, &roied_roi_rectangle, 0, &cropped_map_id, &cropped_imgpatch, &cropped_ptr,
                                       VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    vx_map_id out_map_id;
    vx_imagepatch_addressing_t out_imgpatch;
    void *out_ptr;
    vx_rectangle_t whole_rectangle = {.start_x = 0, .start_y = 0, .end_x = width, .end_y = height};
    ERROR_CHECK_STATUS(vxMapImagePatch(output_img, &whole_rectangle, 0, &out_map_id, &out_imgpatch, &out_ptr,
                                       VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    /*********************************************************/

    vx_coordinates2d_t *coords = NULL;
    vx_map_id map_id;
    vx_size num_items = -1;
    vx_size stride;
    vxQueryArray(coords_arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_items, sizeof(vx_size));
    if (num_items == 2)
    {
        vxMapArrayRange(coords_arr, 0, num_items, &map_id, &stride, (void **)&coords, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);
        for (vx_uint32 i = 0; i < num_items; i++)
        {
            vx_int32 x0 = 0, x1 = roi_height - 1, y0, y1;
            vx_coordinates2d_t p = coords[i];
            vx_float64 tann = tan(p.x * ((3.14) / (NUM_THETAS_LF)));
            vx_float64 shifted = (vx_int32)p.y - NUM_RHOS_HALF;
            vx_float64 coss = cos((vx_int32)p.x * 3.14 / (NUM_THETAS_LF));
            y0 = x0 * tann + SCALING * (shifted / coss);
            y1 = x1 * tann + SCALING * (shifted / coss);

            move_diagonally(cropped_ptr, cropped_imgpatch, roi_width, roi_height, x0, y0, x1, y1);
        }
        vxUnmapArrayRange(coords_arr, map_id);
        vxTruncateArray(coords_arr, 0);
    }

    ERROR_CHECK_STATUS(vxUnmapImagePatch(cropped_in_img, cropped_map_id));
    ERROR_CHECK_STATUS(vxReleaseImage(&cropped_in_img));
    vxCopyImagePatch(input_img, &whole_rectangle, 0, &out_imgpatch, out_ptr, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    ERROR_CHECK_STATUS(vxUnmapImagePatch(output_img, out_map_id));
    ERROR_CHECK_STATUS(vxUnmapArrayRange(roi_arr, roi_mapid));
    // ERROR_CHECK_STATUS(vxReleaseArray(&roi_arr));
    // ERROR_CHECK_STATUS(vxReleaseArray(&coords_arr));
    // ERROR_CHECK_STATUS(vxReleaseImage(&input_img));
    // ERROR_CHECK_STATUS(vxReleaseImage(&output_img));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK initialization(vx_node node, const vx_reference *parameters, vx_uint32 num)
{

    // printf("intialization\n\n");
    vx_array cos_lut = (vx_array)parameters[2];
    vx_array sin_lut = (vx_array)parameters[3];

    for (vx_uint32 i = 0; i < NUM_THETAS; i++)
    {
        vx_float64 ptr = cos(i / NUM_THETAS_LF * PI);
        vxAddArrayItems(cos_lut, 1, &ptr, sizeof(vx_float64));
        ptr = sin(i / NUM_THETAS_LF * PI);
        vxAddArrayItems(sin_lut, 1, &ptr, sizeof(vx_float64));
    }
    return VX_SUCCESS;
}

vx_status VX_CALLBACK deinitialization(vx_node node, const vx_reference *parameters, vx_uint32 num)
{

    // printf("deintialization\n\n");
    vx_array cos_lut = (vx_array)parameters[2];
    vx_array sin_lut = (vx_array)parameters[3];

    ERROR_CHECK_STATUS(vxReleaseArray(&cos_lut));
    ERROR_CHECK_STATUS(vxReleaseArray(&sin_lut));
    return VX_SUCCESS;
}
vx_status registerUserKernels(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context,
                                       "app.userkernels.hough_transform",
                                       USER_KERNEL_HOUGH_TRANSFORM,
                                       hough_host_side_function,
                                       4, // numParams
                                       hough_validator,
                                       initialization,
                                       deinitialization);
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED)); // output
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    vxAddLogEntry((vx_reference)context, VX_SUCCESS, "OK: registered user kernel app.userkernels.hough_transform\n");

    kernel = vxAddUserKernel(context,
                             "app.userkernels.coloring",
                             USER_KERNEL_COLORING,
                             coloring_host_side_function,
                             4, // numParams
                             colorNodeValidator,
                             NULL,
                             NULL);
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED)); // output
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return VX_SUCCESS;
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0)
    {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}
int32_t appInit()
{
    int32_t status = 0;

    status = appCommonInit();

    if(status == 0)
    {
        tivxInit();
        tivxHostInit();
    }
    return status;
}

int32_t appDeInit()
{
    int32_t status = 0;

    tivxHostDeInit();
    tivxDeInit();
    appCommonDeInit();

    return status;
}

int main(int argc, char *argv[])
{

    app_perf_point_t performance;
    int status = 0;

    status = appInit();

    if (argc < 2)
    {
        printf("Usage:\n"
               "./SegmentacijaSlike --image <imageName>\n"
               "./cannyDetect --live \n");
        return 0;
    }

    vx_uint32 width = 1280, height = 720;

    vx_context context = vxCreateContext();

    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    ERROR_CHECK_STATUS(registerUserKernels(context));

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);


    /**************************************
     * Reading image from file
     * ************************************/
    char* option = argv[1];
     vx_image input_rgb_image;
     if (!strcmp(option,"--image"))
    {
        struct read_image_attributes attr;
        input_rgb_image = createImageFromFile(context, argv[2], &attr);
        width = attr.width;
        height = attr.height;
    }
    else if(!strcmp(option,"--video")){
        input_rgb_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    }
    ERROR_CHECK_OBJECT(input_rgb_image);
    vx_image output_filtered_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    ERROR_CHECK_OBJECT(output_filtered_image);

    /*****************************************
     * Working ROI
     * ***************************************
     */
    vx_rectangle_t roi_rect = {.start_x = 350, .start_y = 350, .end_x = width - 450, .end_y = height - 230};
    // vx_rectangle_t roi_rect = {.start_x = 0, .start_y = 350, .end_x=width, .end_y=height};
    vx_uint32 roi_width = roi_rect.end_x - roi_rect.start_x;
    vx_uint32 roi_height = roi_rect.end_y - roi_rect.start_y;

    /*****************************************/
    // virtual images creation
    vx_image yuv_image = vxCreateImage(context, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    vx_image gray_roi = vxCreateImageFromROI(luma_image, &roi_rect);
    vx_image blured_image[2];
    for (vx_size i = 0; i < 2; i++)
    {
        blured_image[i] = vxCreateImage(context, roi_width, roi_height, VX_DF_IMAGE_U8);
        ERROR_CHECK_OBJECT(blured_image[i]);
    }
    vx_image edged_image = vxCreateImage(context, roi_width, roi_height, VX_DF_IMAGE_U8);

    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);
    ERROR_CHECK_OBJECT(edged_image);
    // ERROR_CHECK_OBJECT(roied_image_roi);

    vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_int32 lower = 180, upper = 200;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
    ERROR_CHECK_OBJECT(hyst);
    vx_int32 gradient_size = 3;
    vx_int32 maxcount = 2;
    /*user*/
    vx_array lines_user = vxCreateArray(context, VX_TYPE_COORDINATES2D, maxcount);

    vx_node nodes[] =
        {
            vxColorConvertNode(graph, input_rgb_image, yuv_image),
            vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
            vxGaussian3x3Node(graph, gray_roi, blured_image[0]),
            vxGaussian3x3Node(graph, blured_image[0], blured_image[1]),
            vxCannyEdgeDetectorNode(graph, blured_image[1], hyst, gradient_size, VX_NORM_L1, edged_image),
            userHoughTransformNode(graph, edged_image, lines_user),
            userColorNode(graph, input_rgb_image, lines_user, output_filtered_image, roi_rect)};

    for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    clock_t begin,end;
    double time_spent = 0.0;


    if (!strcmp(option,"--image"))
    {
        begin = clock();

        ERROR_CHECK_STATUS(vxProcessGraph(graph));

        end = clock();
        time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
        /*****************************************************
         * Writing image to file
         * ***************************************************/
        printf("prije write_image\n");
        //writeImage(luma_image,"/media/luma_image.ppm");
        //writeImage(blured_image[1],"/media/blured_image[1].ppm");
        writeImage(edged_image,"/media/edged_image.ppm");
        writeImage(output_filtered_image,"/media/finished.ppm");
        /***********************************************************************/

    }
    else if(!strcmp(option,"--video")){ 
        FILE * video_file = fopen(argv[2],"rb");
        FILE* output_file = fopen("/media/output_video.bin","wb");
        if(!video_file || !output_file )
        {
            printf("Neuspjesno otvaranje fajlova\n");
            appDeInit();
            return 0;
        }
        vx_rectangle_t in_rect = {0,0,width,height};
        vx_map_id in_map;
        vx_imagepatch_addressing_t addr;
        void *in_ptr;
        ERROR_CHECK_STATUS(vxMapImagePatch(input_rgb_image,&in_rect,0,&in_map,&addr,&in_ptr,VX_WRITE_ONLY,VX_MEMORY_TYPE_HOST,VX_NOGAP_X));
        vx_int32 count = 0;
        appPerfStatsResetAll();
        while(fread(in_ptr,1,width*height*3,video_file) == width*height*3){
            appPerfPointBegin(&performance);
            ERROR_CHECK_STATUS(vxUnmapImagePatch(input_rgb_image,in_map));

            printf("FRAME count =%d\n",count++);
            ERROR_CHECK_STATUS(vxProcessGraph(graph));
            appPerfPointEnd(&performance);

            ERROR_CHECK_STATUS(vxMapImagePatch(output_filtered_image,&in_rect,0,&in_map,&addr,&in_ptr,VX_WRITE_ONLY,VX_MEMORY_TYPE_HOST,VX_NOGAP_X));
            fwrite(in_ptr,1,width*height*3,output_file);
            ERROR_CHECK_STATUS(vxUnmapImagePatch(output_filtered_image,in_map));

            ERROR_CHECK_STATUS(vxMapImagePatch(input_rgb_image,&in_rect,0,&in_map,&addr,&in_ptr,VX_WRITE_ONLY,VX_MEMORY_TYPE_HOST,VX_NOGAP_X));
        }
        ERROR_CHECK_STATUS(vxUnmapImagePatch(input_rgb_image,in_map));
        fclose(video_file);
        fclose(output_file);
    }
    tivx_utils_graph_perf_print(graph);
    appPerfPointPrintFPS(&performance);
    printf("FPS=%f\n",1.0/time_spent);
    
    ERROR_CHECK_STATUS(vxReleaseImage(&output_filtered_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    
    if(status == 0)
    {
        appDeInit();
    }
    return 0;
}
