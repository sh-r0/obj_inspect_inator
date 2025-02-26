#pragma once
#define GLFW_INCLUDE_VULKAN
#include "glfw3.h"
#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#include "glm/glm.hpp"
#include "glm/gtx/hash.hpp"

#include <filesystem>
#include <array>
#include <vector>
#include <optional>
#include <string>

struct queueFamilyIndices_t {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;

	inline bool isComplete(void) {
		return graphicsFamily.has_value() && presentFamily.has_value();
	}
};

struct swapChainSupportDetails_t {
	VkSurfaceCapabilitiesKHR capabilities;
	std::vector<VkSurfaceFormatKHR> formats;
	std::vector<VkPresentModeKHR> presentModes;
};

struct vertex_t {
	glm::vec3 pos;
	glm::vec3 normal;
	glm::vec2 texCoord;
	uint32_t texId;
	
	static VkVertexInputBindingDescription getBindingDescription() {
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(vertex_t);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions() {
		std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(vertex_t, pos);
		
		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(vertex_t, normal);

		attributeDescriptions[2].binding = 0;
		attributeDescriptions[2].location = 2;
		attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[2].offset = offsetof(vertex_t, texCoord);

		attributeDescriptions[3].binding = 0;
		attributeDescriptions[3].location = 3;
		attributeDescriptions[3].format = VK_FORMAT_R32_UINT;
		attributeDescriptions[3].offset = offsetof(vertex_t, texId);

		return attributeDescriptions;
	}

	bool operator==(const vertex_t& _other) const {
		return pos == _other.pos &&
			texCoord == _other.texCoord &&
			texId == _other.texId &&
			normal == _other.normal;
	}

};

struct uniformBuffer_t {
	alignas(16) glm::mat4 model;
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct imageInfo_t {
	std::string path;
	VkImage image;
	VkDeviceMemory memory;
	VkImageView iView;
	VkSampler sampler;
};

struct renderer_t {
	const char* appName_c = "obj_inspect_inator";
	const uint32_t version_c = 01; 
	const std::vector<const char*> deviceExtensions_c = {
		VK_KHR_SWAPCHAIN_EXTENSION_NAME
	};
	const uint8_t maxFramesInFlight_c = 2;

	GLFWwindow* window;
	uint32_t winSizeX = 1200, winSizeY = 1000;
	bool framebuffResized = false;

	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkSurfaceKHR surface;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	VkQueue graphicsQueue;
	VkQueue presentQueue;
	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;
	VkImage depthImage; VkImageView depthImageView;
	VkDeviceMemory depthImageMemory;
	VkRenderPass renderPass;
	VkDescriptorSetLayout descriptorSetLayout;
	VkPipelineLayout pipelineLayout;
	VkPipeline graphicsPipeline;
	VkCommandPool commandPool;
	VkBuffer vertexBuffer;
	VkDeviceMemory vertexBufferMemory;
	VkBuffer indexBuffer;
	VkDeviceMemory indexBufferMemory;
	uint32_t texturesCount = 0;
	std::array<imageInfo_t,16> textureImages;

	std::vector<vertex_t> vertices = {};
	std::vector<uint32_t> indices = {};
	float rotX = 0, rotYZ = 0;
	float zoom = 1.0f;

	std::vector<VkBuffer> uniformBuffers;
	std::vector<VkDeviceMemory> uniformBuffersMemory;
	std::vector<void*> uniformBuffersMapped;
	VkDescriptorPool descriptorPool;
	std::vector<VkDescriptorSet> descriptorSets;
	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkSemaphore> imageAvailableSemaphores;
	std::vector<VkSemaphore> renderFinishedSemaphores;
	std::vector<VkFence> inFlightFences;
	uint32_t currentFrame = 0;
	std::filesystem::path objPath;

	void initRenderer(const std::string&);
	
	static void framebuffResizeCallback(GLFWwindow*, int32_t, int32_t);
	void initWindow(void);
	void initVk(void);
	void createWinSurface(void);

	queueFamilyIndices_t findQueueFamilies(VkPhysicalDevice _device);
	bool checkDeviceExtensionSupport(VkPhysicalDevice _device);
	swapChainSupportDetails_t querySwapChainSupport(VkPhysicalDevice _device);
	bool isDeviceSuitable(VkPhysicalDevice _device);
	void pickPhysicalDevice(void);
	void createLogicalDevice(void);
	void createSwapChain(void);
	uint32_t findMemoryType(uint32_t, VkMemoryPropertyFlags);
	VkImageView createImageView(VkImage, VkFormat, VkImageAspectFlags);

	void createVkInstance(void);
	void setupDebugMessenger(void);
	void createImageViews(void);
	void createRenderPass(void);
	void createDescriptorSetLayout(void);
	VkFormat findSupportedFormat(const std::vector<VkFormat>&, VkImageTiling, VkFormatFeatureFlags);
	VkFormat findDepthFormat(void);
	void createDepthResources(void);
	void createGraphicsPipeline(void);
	void createFramebuffers(void);
	void createImage(uint32_t, uint32_t, VkFormat, VkImageTiling, VkImageUsageFlags, VkMemoryPropertyFlags, VkImage& , VkDeviceMemory&);
	void transitionImageLayout(VkImage, VkFormat, VkImageLayout, VkImageLayout);
	void copyBufferToImage(VkBuffer, VkImage, uint32_t, uint32_t);
	void createTextureImages(std::string&, VkImage&, VkDeviceMemory&);
	void createTextureImageViews(VkImage&, VkImageView&);
	void createTextureSamplers(VkSampler&);
	void createCommandPool(void);
	VkCommandBuffer beginSingleTimeCommands(void);
	void endSingleTimeCommands(VkCommandBuffer);
	void copyBuffer(VkBuffer, VkBuffer, VkDeviceSize);
	void createBuffer(VkDeviceSize, VkBufferUsageFlags, VkMemoryPropertyFlags, VkBuffer&, VkDeviceMemory&);
	void createVertexBuffer(void);
	void createIndexBuffer(void);
	void createUniformBuffers(void);
	void createDescriptorPool(void);
	void createDescriptorSets(void);
	void createCommandBuffers(void);
	void createSyncObjects(void);
	
	void recreateSwapChain(void);
	void recordCommandBuffer(VkCommandBuffer, uint32_t);

	void loadObj();
	void updateUBO(void);
	void drawFrame(void);

	void cleanupSwapChain(void);
	void cleanup(void);
};
