#include <iostream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <sstream>
#include <thread>
#include <chrono>

#include <random>

#include <gpgpu.h>

constexpr int32_t kWidth = 1024;
constexpr int32_t kHeight = 1024;

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
}

void main() {
	GetGPGPUInfo();
	
	if (!glfwInit()) {
		glfwTerminate();
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
	GLFWwindow* window = glfwCreateWindow(1024, 1024, "ISIMA_PROJECT", nullptr, nullptr);
	if (window) {
		glfwMakeContextCurrent(window);
		gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glDisable(GL_DEPTH_TEST);

		cudaResourceDesc cuda_resource_desc;
		memset(&cuda_resource_desc, 0, sizeof(cuda_resource_desc));
		cuda_resource_desc.resType = cudaResourceTypeArray;

		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
		
		cudaArray_t array_map;
		cudaMallocArray(&array_map, &channelDesc, kWidth, kHeight, cudaArraySurfaceLoadStore);
		cuda_resource_desc.res.array.array = array_map;
		cudaSurfaceObject_t surface_map = 0;
		cudaCreateSurfaceObject(&surface_map, &cuda_resource_desc);

		GLuint texture;
		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 1024, 1024, 0, GL_RGBA, GL_FLOAT, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		cudaGLSetGLDevice(0);
		cudaGraphicsResource_t cuda_graphic_resource;
		cudaGraphicsGLRegisterImage(&cuda_graphic_resource, texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);

		GLuint fbo = 0;
		glGenFramebuffers(1, &fbo);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);

		auto MapSurface = [](cudaGraphicsResource_t& cuda_resource) -> cudaSurfaceObject_t {
			cudaGraphicsMapResources(1, &cuda_resource);
			cudaArray_t writeArray;
			cudaGraphicsSubResourceGetMappedArray(&writeArray, cuda_resource, 0, 0);
			cudaResourceDesc wdsc;
			wdsc.resType = cudaResourceTypeArray;
			wdsc.res.array.array = writeArray;
			cudaSurfaceObject_t surface;
			cudaCreateSurfaceObject(&surface, &wdsc);
			return surface;
		};
		auto UnmapSurface = [](cudaGraphicsResource_t& cuda_resource, cudaSurfaceObject_t& surface) {
			cudaDestroySurfaceObject(surface);
			cudaGraphicsUnmapResources(1, &cuda_resource);
			cudaStreamSynchronize(0);
		};

		glfwSwapInterval(1);
		glfwSetKeyCallback(window, key_callback);
		int32_t current_frame = 0;
		while (!glfwWindowShouldClose(window)) {
			glfwPollEvents();
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			// START MAIN LOOP
			DrawUVs(surface_map, kWidth, kHeight, static_cast<float>(current_frame)*0.01f);
			// END MAIN LOOP

			cudaGraphicsMapResources(1, &cuda_graphic_resource);
			cudaArray_t array_OpenGL;
			cudaGraphicsSubResourceGetMappedArray(&array_OpenGL, cuda_graphic_resource, 0, 0);
			cuda_resource_desc.res.array.array = array_OpenGL;
			cudaSurfaceObject_t surface_OpenGL;
			cudaCreateSurfaceObject(&surface_OpenGL, &cuda_resource_desc);
			CopyTo(surface_map, surface_OpenGL, kWidth, kHeight);
			cudaDestroySurfaceObject(surface_OpenGL);
			cudaGraphicsUnmapResources(1, &cuda_graphic_resource);
			cudaStreamSynchronize(0);

			glViewport(0, 0, width, height);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
			glBlitFramebuffer(
				0, 0, kWidth, kHeight,
				0, 0, width, height,
				GL_COLOR_BUFFER_BIT, GL_LINEAR);
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glfwSwapBuffers(window);
			++current_frame;
		}
		glfwDestroyWindow(window);
	}
	glfwTerminate();
}
