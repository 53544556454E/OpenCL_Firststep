#include <algorithm>
#include <format>
#include <fstream>
#include <iostream>
#include <random>

#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/opencl.hpp>

const cl_uint kWidth = 1920;
const cl_uint kHeight = 1080;
const cl_uint kThreads = kWidth * kHeight;
const cl_uint kPlanePixels = kThreads;
const cl_uint kPlaneBytes = sizeof (float) * kPlanePixels;

void FillRandomUints(cl_uint *dst, size_t uints);
void OutputImage(cl_uint width, cl_uint height, const float *red, const float *green, const float *blue);

int main()
{
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	platforms.at(0).getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context ctx(devices.at(0));
	cl::CommandQueue command_queue(ctx);

	std::string program_string;
	{
		std::ifstream ifs("kernel.cl");
		program_string = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	}

	cl::Program program(ctx, program_string, true);
	cl::Kernel random_test_kernel(program, "RandomTest");

	cl::Buffer picture_buffer(ctx, CL_MEM_READ_WRITE, kPlaneBytes * 3);
	cl::Buffer random_buffer(ctx, CL_MEM_READ_WRITE| CL_MEM_HOST_WRITE_ONLY, sizeof (cl_uint) * kThreads);

	{
		void *p = command_queue.enqueueMapBuffer(random_buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof (cl_uint) * kThreads);
		FillRandomUints(static_cast<cl_uint *>(p), kThreads);
		command_queue.enqueueUnmapMemObject(random_buffer, p);
		command_queue.finish();
	}

	command_queue.enqueueFillBuffer(picture_buffer, cl_float(0), 0, kPlaneBytes * 3);
	command_queue.finish();

	random_test_kernel.setArg(0, picture_buffer);
	random_test_kernel.setArg(1, random_buffer);
	random_test_kernel.setArg(2, kPlanePixels);
	command_queue.enqueueNDRangeKernel(random_test_kernel, cl::NDRange(0), cl::NDRange(kThreads));
	command_queue.finish();

	{
		void *p = command_queue.enqueueMapBuffer(picture_buffer, CL_TRUE, CL_MAP_READ, 0, kPlaneBytes * 3);
		float *r = static_cast<float *>(p);
		float *g = r + kPlanePixels;
		float *b = g + kPlanePixels;

		OutputImage(kWidth, kHeight, r, g, b);
		command_queue.enqueueUnmapMemObject(picture_buffer, p);
	}
}

void FillRandomUints(cl_uint *dst, size_t uints)
{
	std::random_device seed_gen;
	std::mt19937 engine(seed_gen());

	while (uints--)
	{
		cl_uint v = engine();

		while (!v)
			v = engine();

		*dst++ = v;
	}
}

void OutputImage(cl_uint width, cl_uint height, const float *red, const float *green, const float *blue)
{
	std::cout << std::format("P3 {} {} 255\n", width, height);

	for (cl_uint y=0; y<height; ++y)
	{
		for (cl_uint x=0; x<width; ++x)
		{
			int r = std::clamp(static_cast<int>(255 * *red++), 0, 255);
			int g = std::clamp(static_cast<int>(255 * *green++), 0, 255);
			int b = std::clamp(static_cast<int>(255 * *blue++), 0, 255);

			std::cout << std::format("{} {} {}\n", r, g, b);
		}
	}

	std::cout.flush();
}
