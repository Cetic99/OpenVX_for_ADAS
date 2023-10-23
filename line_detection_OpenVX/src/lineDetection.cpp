#include <VX/vx.h>
#include <VX/vx_compatibility.h>
#include "opencv2/opencv.hpp"
#include <string>
#include <VX/vx_intel_volatile.h>
#include <VX/vx_types.h>
#include <VX/vxu.h>

using namespace cv;
using namespace std;

#define NUM_THETAS      (500)
#define NUM_THETAS_LF   (500.0)
#define NUM_RHOS        (500)
#define SCALING         (10.0)

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
}

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
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
    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e
{
    USER_KERNEL_HOUGH_TRANSFORM     = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x001,
};

////////
// The node creation interface for the "app.userkernels.median_blur" kernel.
// This user kernel example expects parameters in the following order:
//   parameter #0  --  input image  of format VX_DF_IMAGE_U8
//   parameter #1  --  output image of format VX_DF_IMAGE_U8
//   parameter #2  --  scalar ksize of type   VX_TYPE_INT32
//
// TODO:********
//   1. Use vxGetKernelByEnum API to get a kernel object from USER_KERNEL_MEDIAN_BLUR.
//      Note that you need to use vxGetContext API to get the context from a graph object.
//   2. Use vxCreateGenericNode API to create a node from the kernel object.
//   3. Create scalar objects for ksize parameter.
//   4. Use vxSetParameterByIndex API to set node arguments.
//   5. Release the kernel and scalar objects that are not needed any more.
//   6. Use ERROR_CHECK_OBJECT and ERROR_CHECK_STATUS macros for error detection.
vx_node userHoughTransformNode( vx_graph graph,
                            vx_image input,
                            vx_array rects)
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel   = vxGetKernelByEnum( context, USER_KERNEL_HOUGH_TRANSFORM );
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) rects ) );

    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

////////
// The user kernel validator callback should check to make sure that all the input
// parameters have correct data types and set meta format for the output parameters.
// The input parameters to be validated are:
//   parameter #0  --  input image  of format VX_DF_IMAGE_U8
// The output parameters that requires setting meta format is:
//   parameter #1  --  output image of format VX_DF_IMAGE_U8 with the same dimensions as input
// TODO:********
//   1. Check to make sure that the input image format is VX_DF_FORMAT_U8.
//   2. Check to make sure that the scalar type is VX_TYPE_INT32.
//   3. Query the input image for the image dimensions and set the output image
//      meta data to have the same dimensions as input and VX_DF_FORMAT_U8.
vx_status VX_CALLBACK hough_validator( vx_node node,
                                             const vx_reference parameters[], vx_uint32 num,
                                             vx_meta_format metas[] )
{
    // parameter #0 -- input image of format VX_DF_IMAGE_U8
    vx_df_image format = VX_DF_IMAGE_VIRT;
    ERROR_CHECK_STATUS( vxQueryImage( ( vx_image )parameters[0], VX_IMAGE_FORMAT, &format, sizeof( format ) ) );
    if( format != VX_DF_IMAGE_U8 )
    {
        return VX_ERROR_INVALID_FORMAT;
    }

    // parameter #1 -- output array should be of itemtype VX_TYPE_RECTANGLE
    format = VX_TYPE_RECTANGLE;
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_ARRAY_ITEMTYPE, &format, sizeof( format ) ) );

    return VX_SUCCESS;
}

////////
// User kernel host side function gets called to execute the user kernel node.
// You need to wrap the OpenVX objects into OpenCV Mat objects and call cv::medianBlur.
//
// TODO:********
//   1. Get ksize value from scalar object in refs[2].
//   2. Access input and output image patches and wrap it in a cv::Mat object.
//      Use the cv::Mat mat(width, height, CV_U8, ptr, addr.stride_y) to wrap
//      an OpenVX image plane buffer in an OpenCV mat object. Note that
//      you need to access input image with VX_READ_ONLY and output image
//      with VX_WRITE_ONLY using vxMapImagePatch API.
//   3. Just call cv::medianBlur(input, output, ksize)
//   4. Use vxUnmapImagePatch API to give the image buffers control back to OpenVX framework
vx_status VX_CALLBACK hough_host_side_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
    vx_image  input   = ( vx_image ) refs[0];
    vx_array  output  = ( vx_array ) refs[1];

    vx_context context = vxGetContext( ( vx_reference ) node );

    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS( vxQueryImage( input, VX_IMAGE_WIDTH,  &width,  sizeof( width ) ) );
    ERROR_CHECK_STATUS( vxQueryImage( input, VX_IMAGE_HEIGHT, &height, sizeof( height ) ) );

    vx_rectangle_t rect = { 0, 0, width, height };
    vx_map_id map_input;
    vx_imagepatch_addressing_t addr_input;
    void * ptr_input;
    ERROR_CHECK_STATUS( vxMapImagePatch( input,  &rect, 0, &map_input, &addr_input,  &ptr_input,  VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );

    //vx_matrix accum = vxCreateMatrix(context,VX_TYPE_UINT16,NUM_THETAS,NUM_RHOS);
    //ERROR_CHECK_OBJECT(accum);

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
    vx_image accum = vxCreateImage(context,NUM_RHOS,NUM_THETAS,VX_DF_IMAGE_U16);
    ERROR_CHECK_STATUS(vxMapImagePatch(accum, &rect_transformed, 0, &map_id,
                                &addr, &base_ptr,
                                VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, 0));
                                
    for (vx_uint32 i = 0; i < addr.dim_x*addr.dim_y; i++) {
        vx_uint16 *ptr2 = (vx_uint16*)vxFormatImagePatchAddress1d(base_ptr, i, &addr);
        *ptr2 = 0;
    }
    /*****************************************************************************/

    /********************************************
     * Creating trigonometric LUT
     * ******************************************/

    // vx_lut trig_lut = vxCreateLUT(context,VX_TYPE_FLOAT32, NUM_THETAS);
    // for(vx_uint32 i = 0; i<NUM_THETAS;i++){
        
    // }
    

    //vx_uint16 (*transformisano_arr)[NUM_RHOS] = (vx_uint16(*)[NUM_RHOS])base_ptr;

    //vx_uint16 (*transformisano_arr)[NUM_RHOS] = (vx_uint16(*)[NUM_RHOS])calloc(NUM_THETAS * NUM_RHOS, sizeof(vx_uint16));

    for (vx_uint32 i = 0; i < width; i++) {
		for (vx_uint32 j = 0; j < height; j++) {
            vx_uint8 *ptr2 = (vx_uint8*)vxFormatImagePatchAddress2d(ptr_input, i, j, &addr_input);
            
			//1. calculate variables for non-zero pixels
            if (*ptr2 != 0) {
				// do calculation
				for (vx_uint32 k = 0; k < NUM_THETAS; k++) {

					vx_uint32 rho =  (NUM_RHOS/2)+ (j * cos(k / NUM_THETAS_LF * CV_PI) - i * sin(k / NUM_THETAS_LF * CV_PI))/SCALING;
					//transformisano_arr[k][rho]++;
                    vx_uint16 *ptr2 = (vx_uint16*)vxFormatImagePatchAddress2d(base_ptr, rho, k, &addr);
                    *ptr2 = *ptr2 + 1 ;
                    //transformisano_arr[k][rho]++;
				}
			}
		}
	}

    // Mat m(NUM_THETAS, NUM_RHOS, CV_16U, Scalar(0));

	// for (int i = 0; i < width; i++) {
	// 	for (int j = 0; j < height; j++) {
    //         vx_uint8 *ptr2 = (vx_uint8*)vxFormatImagePatchAddress2d(ptr_input,
    //                                                           i, j, &addr_input);
	// 		if (*ptr2 != 0) {
	// 			// radi transformaciju
	// 			for (int k = 0; k < NUM_THETAS; k++) {
	// 				int rho =  (NUM_RHOS/2)+ (j * cos(k / NUM_THETAS_LF * CV_PI) - i * sin(k / NUM_THETAS_LF * CV_PI))/SCALING;

	// 				m.at<ushort>(k, rho)++;
	// 			}
	// 		}
	// 	}
	// }


    /****************************************************
     *      Can be commented
     * **************************************************/
    //Mat m(NUM_THETAS,NUM_RHOS,CV_16U,transformisano_arr);

    Mat t( NUM_THETAS, NUM_RHOS, CV_16U, base_ptr, addr.stride_y );
    Mat nrm;
    normalize(t,nrm,0,255, NORM_MINMAX,CV_8U);
    imshow("houghed",nrm);

    /****************************************************/
    
    ERROR_CHECK_STATUS( vxUnmapImagePatch( input,  map_input ) );
    ERROR_CHECK_STATUS( vxUnmapImagePatch( accum,  map_id ) );
    //free(transformisano_arr);

    return VX_SUCCESS;
}

////////
// User kernels needs to be registered with every OpenVX context before use in a graph.
//
// TODO:********
//   1. Use vxAddUserKernel API to register "app.userkernels.median_blur" with
//      kernel enumeration = USER_KERNEL_MEDIAN_BLUR, numParams = 3, and
//      all of the user kernel callback functions you implemented above.
//   2. Use vxAddParameterToKernel API to specify direction, data_type, and
//      state of all 3 parameters to the kernel. Look into the comments of
//      userMedianBlurNode function (above) to details about the order of
//      kernel parameters and their types.
//   3. Use vxFinalizeKernel API to make the kernel ready to use in a graph.
//      Note that the kernel object is still valid after this call.
//      So you need to call vxReleaseKernel before returning from this function.
vx_status registerUserKernel( vx_context context )
{
    vx_kernel kernel = vxAddUserKernel( context,
                                    "app.userkernels.hough_transform",
                                    USER_KERNEL_HOUGH_TRANSFORM,
                                    hough_host_side_function,
                                    2,   // numParams
                                    hough_validator,
                                    NULL,
                                    NULL );
    ERROR_CHECK_OBJECT( kernel );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // output
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );
    

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.hough_transform\n" );
    return VX_SUCCESS;
}

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
    size_t len = strlen(string);
    if (len > 0) {
        printf("%s", string);
        if (string[len - 1] != '\n')
            printf("\n");
        fflush(stdout);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) {
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
    ERROR_CHECK_STATUS( registerUserKernel( context ) );
    
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);
    
    // context data
    vx_image input_rgb_image        = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image output_filtered_image  = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_filtered_image);


    // virtual images creation
    vx_image yuv_image          = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_IYUV);
    vx_image luma_image         = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    vx_size num_blures = 1;
    vx_image blured_image[num_blures];
        for(vx_size i = 0; i< num_blures;i++){
            blured_image[i] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        }
    vx_image edged_image        = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    

    ERROR_CHECK_OBJECT(yuv_image);
    ERROR_CHECK_OBJECT(luma_image);
    ERROR_CHECK_OBJECT(edged_image);

    vx_threshold hyst = vxCreateThreshold(context, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_int32 lower = 150, upper = 200;
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_LOWER, &lower, sizeof(lower));
    vxSetThresholdAttribute(hyst, VX_THRESHOLD_ATTRIBUTE_THRESHOLD_UPPER, &upper, sizeof(upper));
    ERROR_CHECK_OBJECT(hyst);
    vx_int32 gradient_size = 3;
    vx_float32 deltaRho = 1;
    vx_float32 deltaTheta = 3.14/180.0;
    vx_int32 thr_val = 100;
    vx_int32 lineLen = 50;
    vx_int32 lineGap = 1;
    vx_int32 maxcount = 30;
    vx_array lines = vxCreateArray(context,VX_TYPE_RECTANGLE,maxcount);
    vx_int32 count_var = 0;
    vx_scalar count = vxCreateScalar(context,VX_TYPE_INT32,&count_var);
    /*user*/
    vx_array lines_user = vxCreateArray(context,VX_TYPE_RECTANGLE,maxcount);

    vx_node nodes[] =
    {
        vxColorConvertNode(graph, input_rgb_image, yuv_image),
        vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, luma_image),
        //vxCannyEdgeDetectorNode(graph, luma_image, hyst, gradient_size, VX_NORM_L1, output_filtered_image),
        vxGaussian3x3Node(graph,luma_image,blured_image[0]),
        vxCannyEdgeDetectorNode(graph, blured_image[0], hyst, gradient_size, VX_NORM_L1, output_filtered_image),
        vxHoughLinesPNodeIntel(graph, output_filtered_image,deltaRho,deltaTheta,thr_val,lineLen,lineGap,maxcount,lines,count),
        /*used for userhough*/
        vxCannyEdgeDetectorNode(graph, blured_image[0], hyst, gradient_size, VX_NORM_L1, edged_image),
        userHoughTransformNode(graph,edged_image,lines_user)
    };

    for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodes[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
    }
    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );
    
    string option = argv[1];
    Mat input;

    if (option == "--image")
    {
        string imageLocation = argv[2];
        input = imread(imageLocation.c_str());
        if (input.empty()) {
           printf("Image not found\n");
           return 0;
        }
        resize(input, input, Size(width, height));
        imshow("inputWindow", input);
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout{};
        cv_rgb_image_layout.dim_x = input.cols;
        cv_rgb_image_layout.dim_y = input.rows;
        cv_rgb_image_layout.stride_x = input.elemSize();
        cv_rgb_image_layout.stride_y = input.step;
        vx_uint8 * cv_rgb_image_buffer = input.data;
        ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                            &cv_rgb_image_layout, cv_rgb_image_buffer,
                                            VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );
        ERROR_CHECK_STATUS( vxProcessGraph( graph ) );
        vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;
        void * ptr;
        ERROR_CHECK_STATUS( vxMapImagePatch( output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
                                            VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
        Mat mat( height, width, CV_8U, ptr, addr.stride_y );
        imshow( "CannyDetect", mat );
        waitKey(0);
        ERROR_CHECK_STATUS( vxUnmapImagePatch( output_filtered_image, map_id ) );
    }
    else if (option == "--live")
    {
        VideoCapture cap("../videos/working_video.mp4");
        if (!cap.isOpened()) {
            printf("Unable to open camera\n");
            return 0;
        }
        for(;;) {
            cap >> input;
            resize(input, input, Size(width, height));
            if(waitKey(30) >= 0) break;
            vx_rectangle_t cv_rgb_image_region;
            cv_rgb_image_region.start_x    = 0;
            cv_rgb_image_region.start_y    = 0;
            cv_rgb_image_region.end_x      = width;
            cv_rgb_image_region.end_y      = height;
            vx_imagepatch_addressing_t cv_rgb_image_layout;
            cv_rgb_image_layout.dim_x = input.cols;
            cv_rgb_image_layout.dim_y = input.rows;
            cv_rgb_image_layout.stride_x   = input.elemSize();
            cv_rgb_image_layout.stride_y   = input.step;
            vx_uint8 * cv_rgb_image_buffer = input.data;
            ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                                &cv_rgb_image_layout, cv_rgb_image_buffer,
                                                VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );
            ERROR_CHECK_STATUS( vxProcessGraph( graph ) );
            vx_rectangle_t rect = { 0, 0, (vx_uint32)width, (vx_uint32)height };
            vx_map_id map_id;
            vx_imagepatch_addressing_t addr;
            void * ptr;
            ERROR_CHECK_STATUS( vxMapImagePatch( output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
                                            VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
            Mat mat( height, width, CV_8U, ptr, addr.stride_y );

            /*=================== drawing lines =============================================*/
            vx_size stride = sizeof(vx_rectangle_t);
            void * line_points_base = NULL;
            vx_int32 count_lines;
            ERROR_CHECK_STATUS(vxReadScalarValue(count,&count_lines));
            if(count_lines>1){
                vxAccessArrayRange(lines, 0, count_lines-1, &stride, &line_points_base,VX_READ_AND_WRITE);
                for(vx_size i = 0; i< count_lines;i++){
                    vx_rectangle_t rect = vxArrayItem(vx_rectangle_t,line_points_base,i,stride);
                    line(input,Point(rect.start_x,rect.start_y),Point(rect.end_x,rect.end_y),Scalar(0,0,255),3);
               }
                vxCommitArrayRange(lines,0,count_lines-1,line_points_base);
            }
            
            /*============================================================================*/
            //imshow( "CannyDetect", mat );
            //imshow("inputWindow", input);
            if(waitKey(30) >= 0) break;
            ERROR_CHECK_STATUS( vxUnmapImagePatch( output_filtered_image, map_id ) );
        }
    }
    else {
        printf("Usage:\n"
                "./cannyDetect --image <imageName>\n"
                "./cannyDetect --live \n");
        return 0;
    }

    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &luma_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &output_filtered_image ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );
    return 0;
}
