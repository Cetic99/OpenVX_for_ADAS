#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "opencv2/opencv.hpp"
#include <string>
#include <VX/vx_intel_volatile.h>
#include <VX/vx_types.h>
#include <VX/vxu.h>

using namespace cv;
using namespace std;

#define NUM_THETAS      (800)
#define NUM_THETAS_LF   (800.0)
#define NUM_RHOS        (800)
#define NUM_RHOS_HALF   (400)
#define SCALING         (16.0)

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
    USER_KERNEL_HOUGH_TRANSFORM     = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE) + 0x001,
    USER_KERNEL_COLORING            = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE) + 0x002,
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

    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)rects));

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

    vx_context context = vxGetContext((vx_reference)node);

    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(height)));

    vx_rectangle_t rect = {0, 0, width, height};
    vx_map_id map_input;
    vx_imagepatch_addressing_t addr_input;
    void *ptr_input;
    ERROR_CHECK_STATUS(vxMapImagePatch(input, &rect, 0, &map_input, &addr_input, &ptr_input, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

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
    vx_image accum_u8 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);

    //dilatation images
    vx_image dilated_0 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image dilated_1 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image dilated_2 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image dilated_3 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);

    //erosion images
    vx_image eroded_0 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image eroded_1 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image eroded_2 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);
    vx_image eroded_3 = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);

    // thresholded image
    vx_image thresholded = vxCreateImage(context, NUM_RHOS, NUM_THETAS, VX_DF_IMAGE_U8);

    ERROR_CHECK_STATUS(vxMapImagePatch(accum, &rect_transformed, 0, &map_id,
                                       &addr, &base_ptr,
                                       VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));

    for (vx_uint32 i = 0; i < addr.dim_x * addr.dim_y; i++)
    {
        vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress1d(base_ptr, i, &addr);
        *ptr2 = 0;
    }
    /*****************************************************************************/

    /********************************************
     * Creating trigonometric LUT
     * ******************************************/

    vx_array cos_lut = vxCreateArray(context,VX_TYPE_FLOAT32, NUM_THETAS);
    vx_array sin_lut = vxCreateArray(context,VX_TYPE_FLOAT32, NUM_THETAS);

    for(vx_uint32 i = 0; i<NUM_THETAS;i++){
        vx_float32 ptr = cos(i / NUM_THETAS_LF * CV_PI);
        vxAddArrayItems(cos_lut,1,&ptr,sizeof(vx_float32));
        ptr = sin(i / NUM_THETAS_LF * CV_PI);
        vxAddArrayItems(sin_lut,1,&ptr,sizeof(vx_float32));
    }
     vx_map_id cos_lut_map_id;
     void *cos_lut_base_ptr = NULL;
     vx_size stride;
     ERROR_CHECK_STATUS(vxMapArrayRange(cos_lut,0,NUM_THETAS,&cos_lut_map_id,&stride,&cos_lut_base_ptr,VX_READ_AND_WRITE,VX_MEMORY_TYPE_HOST,0));
     vx_float32 *trig_cos = (vx_float32*)cos_lut_base_ptr;

     
     vx_map_id sin_lut_map_id;
     void *sin_lut_base_ptr = NULL;
     ERROR_CHECK_STATUS(vxMapArrayRange(sin_lut,0,NUM_THETAS,&sin_lut_map_id,&stride,&sin_lut_base_ptr,VX_READ_AND_WRITE,VX_MEMORY_TYPE_HOST,0));
     vx_float32 *trig_sin = (vx_float32*)sin_lut_base_ptr;


    for (vx_uint32 i = 0; i < width; i++)
    {
        for (vx_uint32 j = 0; j < height; j++)
        {
            vx_uint8 *ptr2 = (vx_uint8 *)vxFormatImagePatchAddress2d(ptr_input, i, j, &addr_input);

            // 1. calculate variables for non-zero pixels
            if (*ptr2 != 0)
            {
                // do calculation
                for (vx_uint32 k = 0; k < NUM_THETAS; k++)
                {

                    vx_uint32 rho = NUM_RHOS_HALF + (i * trig_cos[k] - j * trig_sin[k]) / SCALING;
                    vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress2d(base_ptr, rho, k, &addr);
                    *ptr2 = *ptr2 + 1;
                    // transformisano_arr[k][rho]++;
                }
            }
        }
    }
    
    ERROR_CHECK_STATUS(vxUnmapArrayRange(cos_lut,cos_lut_map_id));
    ERROR_CHECK_STATUS(vxUnmapArrayRange(sin_lut,sin_lut_map_id));

    ERROR_CHECK_STATUS(vxReleaseArray(&cos_lut));
    ERROR_CHECK_STATUS(vxReleaseArray(&sin_lut));

    /***********************************************
     * Normalization
     * *********************************************/
    vx_int16 max = 0;
    for (vx_uint32 i = 0; i < addr.dim_x * addr.dim_y; i++)
    {
        vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress1d(base_ptr, i, &addr);
        if(*ptr2 > max){
            max = *ptr2;
        }
    }

    for (vx_uint32 i = 0; i < addr.dim_x * addr.dim_y; i++)
    {
        vx_int16 *ptr2 = (vx_int16 *)vxFormatImagePatchAddress1d(base_ptr, i, &addr);
        *ptr2 = *ptr2 * 255/max;
    }
    // Mat t1(NUM_THETAS, NUM_RHOS, CV_16U, base_ptr, addr.stride_y);
    // Mat nrm;
    // normalize(t1, nrm, 0, 255, NORM_MINMAX, CV_8U);
    
    // //Mat nrm;
    // imshow("houghed_real", nrm);
    ERROR_CHECK_STATUS(vxUnmapImagePatch(accum, map_id));

    /************************************************/

    ERROR_CHECK_STATUS(vxuConvertDepth(context,accum,accum_u8,VX_CONVERT_POLICY_WRAP,0));
    ERROR_CHECK_STATUS(vxuDilate3x3(context,accum_u8,dilated_0));
    ERROR_CHECK_STATUS(vxuDilate3x3(context,dilated_0,dilated_1));
    ERROR_CHECK_STATUS(vxuDilate3x3(context,dilated_1,dilated_2));
    ERROR_CHECK_STATUS(vxuDilate3x3(context,dilated_2,dilated_3));
    ERROR_CHECK_STATUS(vxuErode3x3(context,dilated_3,eroded_0));
    ERROR_CHECK_STATUS(vxuErode3x3(context,eroded_0,eroded_1));
    ERROR_CHECK_STATUS(vxuErode3x3(context,eroded_1,eroded_2));
    ERROR_CHECK_STATUS(vxuErode3x3(context,eroded_2,eroded_3));

    /************************************************
     * Thresholding
     * **********************************************/
    vx_threshold img_trh = vxCreateThreshold(context,VX_THRESHOLD_TYPE_BINARY,VX_TYPE_UINT8);
    vx_uint32 true_val = 210;
    vxSetThresholdAttribute(img_trh,VX_THRESHOLD_ATTRIBUTE_THRESHOLD_VALUE ,&true_val,sizeof(true_val));
    true_val = 255;
    vxSetThresholdAttribute(img_trh,VX_THRESHOLD_ATTRIBUTE_TRUE_VALUE ,&true_val,sizeof(true_val));
    true_val = 0;
    vxSetThresholdAttribute(img_trh,VX_THRESHOLD_ATTRIBUTE_FALSE_VALUE ,&true_val,sizeof(true_val));

    vxuThreshold(context,eroded_3,img_trh,thresholded);
    vxReleaseThreshold(&img_trh);

    /****************************************************
     *  Finding peaks
     * **************************************************/
    vx_rectangle_t thresholded_rect = {.start_x = 0, .start_y = 0, .end_x = NUM_RHOS, .end_y = NUM_THETAS};
    vx_map_id thresholded_map_id;
    vx_imagepatch_addressing_t thresholded_img_patch;
    void * thresholded_ptr;
    ERROR_CHECK_STATUS(vxMapImagePatch(thresholded, &thresholded_rect, 0, &thresholded_map_id,
                                       &thresholded_img_patch, &thresholded_ptr,
                                       VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));

    vx_bool line1 = vx_false_e,line2 = vx_false_e;
    vx_bool notfound = vx_true_e;
	for (vx_uint32 i = 0; i < NUM_THETAS && notfound; i++) {
		for (vx_uint32 j = 0; j < NUM_RHOS && notfound; j++) {
            vx_uint8 *pxl = (vx_uint8*)vxFormatImagePatchAddress2d(thresholded_ptr,j,i,&thresholded_img_patch);
			if (*pxl != 0) {
                vx_float32 theta = i * CV_PI / NUM_THETAS_LF;
                vx_float32 tan_theta = tan(theta);
                if(sin(theta) > 0.9 )
                    continue;
                if(tan_theta>0 && line1 == vx_false_e){
                    line1 = vx_true_e;
                    vx_coordinates2d_t coord = {.x = i, .y=j};
                    //cout<<"x="<<j<<", y="<<i<<std::endl;
					vxAddArrayItems(output_arr,1,&coord,sizeof(vx_coordinates2d_t));
                }
                else if(tan_theta<0 && line2 == vx_false_e){
                    line2 = vx_true_e;
                    vx_coordinates2d_t coord = {.x = i, .y=j};
                    //cout<<"x="<<j<<", y="<<i<<std::endl;
					vxAddArrayItems(output_arr,1,&coord,sizeof(vx_coordinates2d_t));
                }

                if(line2 == vx_true_e && line1 == vx_true_e){
                    notfound = vx_false_e;
                }
					
				
			}
		}
	}
    /******************************************************/

    /******************************************************
     *  Creating lines
     * ****************************************************/
    


    ERROR_CHECK_STATUS(vxUnmapImagePatch(thresholded, thresholded_map_id));
    /****************************************************
     *      Can be commented
     * **************************************************/
    ERROR_CHECK_STATUS(vxMapImagePatch(thresholded, &rect_transformed, 0, &map_id,
                                       &addr, &base_ptr,
                                       VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));
     //Mat m(NUM_THETAS,NUM_RHOS,CV_8U,transformisano_arr);

    // Mat t(NUM_THETAS, NUM_RHOS, CV_8U, base_ptr, addr.stride_y);
    // //Mat nrm;
    // //normalize(t, nrm, 0, 255, NORM_MINMAX, CV_8U);
    // imshow("houghed", t);
    ERROR_CHECK_STATUS(vxUnmapImagePatch(thresholded, map_id));
    /****************************************************/

    ERROR_CHECK_STATUS(vxUnmapImagePatch(input, map_input));
    
    // free(transformisano_arr);

    return VX_SUCCESS;
}

vx_node userColorNode(vx_graph graph,
                               vx_image input,
                               vx_array rects,
                               vx_image output)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_COLORING);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);

    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)rects));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference)output));

    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return node;
}
vx_status VX_CALLBACK colorNodeValidator(vx_node node,
                                      const vx_reference parameters[], vx_uint32 num,
                                      vx_meta_format metas[])
{
    // parameter #0 -- input image of format VX_DF_IMAGE_U8
    vx_df_image format = VX_DF_IMAGE_VIRT;
    vx_uint32 in_width,in_height;
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
    return VX_SUCCESS;
}

vx_status VX_CALLBACK coloring_host_side_function(vx_node node, const vx_reference *refs, vx_uint32 num)
{
    vx_image input_img = (vx_image)refs[0];
    vx_array coords_arr = (vx_array)refs[1];
    vx_image output_img = (vx_image)refs[2];

    void *in_ptr;
    vx_uint32 width,height;
    ERROR_CHECK_STATUS(vxQueryImage(input_img,VX_IMAGE_ATTRIBUTE_WIDTH,&width,sizeof(vx_uint32)));
    ERROR_CHECK_STATUS(vxQueryImage(input_img,VX_IMAGE_ATTRIBUTE_HEIGHT,&height,sizeof(vx_uint32)));

    vx_rectangle_t in_rect = {.start_x = 0, .start_y = 0, .end_x = width, .end_y = height };
    vx_map_id in_map_id;
    vx_imagepatch_addressing_t in_imgpatch;
    ERROR_CHECK_STATUS(vxMapImagePatch(input_img, &in_rect, 0, &in_map_id, &in_imgpatch, &in_ptr,
                                          VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));

    void *out_ptr;
    vx_rectangle_t out_rect = {.start_x = 0, .start_y = 0, .end_x = width, .end_y = height };
    vx_map_id out_map_id;
    vx_imagepatch_addressing_t out_imgpatch;
    ERROR_CHECK_STATUS(vxMapImagePatch(output_img, &out_rect, 0, &out_map_id, &out_imgpatch, &out_ptr,
                                          VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));

    Mat mat(height, width, CV_8UC3, in_ptr, in_imgpatch.stride_y);

    vx_coordinates2d_t * coords = NULL;
    vx_map_id map_id;
        vx_size num_items = -1;
        vx_size stride;
        vxQueryArray(coords_arr,VX_ARRAY_ATTRIBUTE_NUMITEMS,&num_items,sizeof(vx_size));
        vxMapArrayRange(coords_arr,0,num_items,&map_id,&stride,(void**)&coords,VX_READ_ONLY,VX_MEMORY_TYPE_HOST,0);
        for (vx_uint32 i = 0; i<num_items; i++) {
            vx_int32 x0 = 400,x1 = 880, y0,y1;
            vx_coordinates2d_t p = coords[i];
            vx_float32 tann = tan(p.x * ((3.14) / (NUM_THETAS_LF)));
            vx_float32 shifted = (vx_int32)p.y - NUM_RHOS_HALF;
            vx_float32 coss =cos((vx_int32)p.x * 3.14 / (NUM_THETAS_LF));
            y0 = x0 * tann + SCALING * (shifted / coss);
            y1 = x1 * tann + SCALING * (shifted / coss);
            line(mat, Point(y0, x0), Point(y1, x1), Scalar(0, 0, 255),3);
        }
        vxUnmapArrayRange(coords_arr,map_id);
        vxTruncateArray(coords_arr,0);

        
        //Mat mat(height, width, CV_8U, ptr, addr.stride_y);
        imshow("CannyDetect", mat);
        ERROR_CHECK_STATUS(vxUnmapImagePatch(input_img, in_map_id));
        ERROR_CHECK_STATUS(vxUnmapImagePatch(output_img, out_map_id));

    return VX_SUCCESS;
}

vx_status registerUserKernels(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context,
                                       "app.userkernels.hough_transform",
                                       USER_KERNEL_HOUGH_TRANSFORM,
                                       hough_host_side_function,
                                       2, // numParams
                                       hough_validator,
                                       NULL,
                                       NULL);
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED)); // output
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    vxAddLogEntry((vx_reference)context, VX_SUCCESS, "OK: registered user kernel app.userkernels.hough_transform\n");

    kernel = vxAddUserKernel(context,
                                       "app.userkernels.coloring",
                                       USER_KERNEL_COLORING,
                                       coloring_host_side_function,
                                       3, // numParams
                                       colorNodeValidator,
                                       NULL,
                                       NULL);
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));  // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED)); // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED));  // output
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

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Usage:\n"
               "./SegmentacijaSlike --image <imageName>\n"
               "./cannyDetect --live \n");
        return 0;
    }

    int width = 1280, height = 720;

    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_false_e);

    ////////
    // Register user kernels with the context.
    //
    // TODO:********
    //   1. Register user kernel with context by calling your implementation of "registerUserKernel()".
    ERROR_CHECK_STATUS(registerUserKernels(context));

    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);

    // context data
    vx_image input_rgb_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image output_filtered_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_filtered_image);

    // virtual images creation
    vx_image yuv_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_size num_blures = 1;
    vx_image blured_image[num_blures];
    for (vx_size i = 0; i < num_blures; i++)
    {
        blured_image[i] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    }
    vx_image edged_image = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);

    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);
    ERROR_CHECK_OBJECT(edged_image);

    vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_int32 lower = 150, upper = 200;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
    ERROR_CHECK_OBJECT(hyst);
    vx_int32 gradient_size = 3;
    vx_int32 maxcount = 200;
    vx_array lines = vxCreateArray(context, VX_TYPE_RECTANGLE, maxcount);
    vx_int32 count_var = 0;
    vx_scalar count = vxCreateScalar(context, VX_TYPE_INT32, &count_var);
    /*user*/
    vx_array lines_user = vxCreateArray(context, VX_TYPE_COORDINATES2D, maxcount);
    

    vx_node nodes[] =
        {
            vxColorConvertNode(graph, input_rgb_image, yuv_image),
            vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
            vxGaussian3x3Node(graph, luma_image, blured_image[0]),
            vxCannyEdgeDetectorNode(graph, blured_image[0], hyst, gradient_size, VX_NORM_L1, edged_image),
            userHoughTransformNode(graph, edged_image, lines_user),
            userColorNode(graph,input_rgb_image,lines_user,output_filtered_image)
        };

    for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
    {
        ERROR_CHECK_OBJECT(nodes[i]);
        ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
    }
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    string option = argv[1];
    Mat input;

    if (option == "--image")
    {
        string imageLocation = argv[2];
        input = imread(imageLocation.c_str());
        if (input.empty())
        {
            printf("Image not found\n");
            return 0;
        }
        resize(input, input, Size(width, height));
        imshow("inputWindow", input);
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x = 0;
        cv_rgb_image_region.start_y = 0;
        cv_rgb_image_region.end_x = width;
        cv_rgb_image_region.end_y = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout{};
        cv_rgb_image_layout.dim_x = input.cols;
        cv_rgb_image_layout.dim_y = input.rows;
        cv_rgb_image_layout.stride_x = input.elemSize();
        cv_rgb_image_layout.stride_y = input.step;
        vx_uint8 *cv_rgb_image_buffer = input.data;
        ERROR_CHECK_STATUS(vxCopyImagePatch(input_rgb_image, &cv_rgb_image_region, 0,
                                            &cv_rgb_image_layout, cv_rgb_image_buffer,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
        ERROR_CHECK_STATUS(vxProcessGraph(graph));

        


        vx_rectangle_t rect = {0, 0, (vx_uint32)width, (vx_uint32)height};
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;

        /***********************************************************************/
        vx_perf_t perfHough = { 0 };
        ERROR_CHECK_STATUS( vxQueryGraph( graph, VX_GRAPH_PERFORMANCE, &perfHough, sizeof( perfHough ) ) );
        printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Hough    %9d %7.3f %7.3f\n",
            ( int )perfHough.num, ( float )perfHough.avg * 1e-6f, ( float )perfHough.min * 1e-6f);

        //void *ptr;
        //ERROR_CHECK_STATUS(vxMapImagePatch(output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
        //                                   VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
        //Mat mat(height, width, CV_8U, ptr, addr.stride_y);
        //imshow("CannyDetect", mat);
        waitKey(0);
        //ERROR_CHECK_STATUS(vxUnmapImagePatch(output_filtered_image, map_id));
    }
    else if (option == "--live")
    {
        VideoCapture cap("../../working_video.mp4");
        if (!cap.isOpened())
        {
            printf("Unable to open camera\n");
            return 0;
        }
        for (;;)
        {
            cap >> input;
            resize(input, input, Size(width, height));
            if (waitKey(30) >= 0)
                break;
            vx_rectangle_t cv_rgb_image_region;
            cv_rgb_image_region.start_x = 0;
            cv_rgb_image_region.start_y = 0;
            cv_rgb_image_region.end_x = width;
            cv_rgb_image_region.end_y = height;
            vx_imagepatch_addressing_t cv_rgb_image_layout;
            cv_rgb_image_layout.dim_x = input.cols;
            cv_rgb_image_layout.dim_y = input.rows;
            cv_rgb_image_layout.stride_x = input.elemSize();
            cv_rgb_image_layout.stride_y = input.step;
            vx_uint8 *cv_rgb_image_buffer = input.data;
            ERROR_CHECK_STATUS(vxCopyImagePatch(input_rgb_image, &cv_rgb_image_region, 0,
                                                &cv_rgb_image_layout, cv_rgb_image_buffer,
                                                VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            ERROR_CHECK_STATUS(vxProcessGraph(graph));


            vx_rectangle_t rect = {0, 0, (vx_uint32)width, (vx_uint32)height};
            vx_map_id map_id;
            vx_imagepatch_addressing_t addr;

        vx_perf_t perfHough = { 0 };
        ERROR_CHECK_STATUS( vxQueryGraph( graph, VX_GRAPH_PERFORMANCE, &perfHough, sizeof( perfHough ) ) );
        printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Hough    %9d %7.3f %7.3f\n",
            ( int )perfHough.num, ( float )perfHough.avg * 1e-6f, ( float )perfHough.min * 1e-6f);
        /***********************************************************************/


            // void *ptr;
            // ERROR_CHECK_STATUS(vxMapImagePatch(output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
            //                                    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X));
            //Mat mat(height, width, CV_8U, ptr, addr.stride_y);

            /*=================== drawing lines =============================================*/
            // vx_size stride = sizeof(vx_rectangle_t);
            // void *line_points_base = NULL;
            // vx_int32 count_lines;
            // ERROR_CHECK_STATUS(vxReadScalarValue(count, &count_lines));
            // if (count_lines > 1)
            // {
            //     vxAccessArrayRange(lines, 0, count_lines - 1, &stride, &line_points_base, VX_READ_AND_WRITE);
            //     for (vx_size i = 0; i < count_lines; i++)
            //     {
            //         vx_rectangle_t rect = vxArrayItem(vx_rectangle_t, line_points_base, i, stride);
            //         line(input, Point(rect.start_x, rect.start_y), Point(rect.end_x, rect.end_y), Scalar(0, 0, 255), 3);
            //     }
            //     vxCommitArrayRange(lines, 0, count_lines - 1, line_points_base);
            // }

            /*============================================================================*/
            // imshow( "CannyDetect", mat );
             //imshow("inputWindow", input);
            if (waitKey(30) >= 0)
                break;
            // ERROR_CHECK_STATUS(vxUnmapImagePatch(output_filtered_image, map_id));
        }
    }
    else
    {
        printf("Usage:\n"
               "./cannyDetect --image <imageName>\n"
               "./cannyDetect --live \n");
        return 0;
    }

    ERROR_CHECK_STATUS(vxReleaseGraph(&graph));
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&luma_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&input_rgb_image));
    ERROR_CHECK_STATUS(vxReleaseImage(&output_filtered_image));
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
    return 0;
}
