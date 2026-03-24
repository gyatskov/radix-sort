/// @file visualize.cpp
/// @brief Vulkan visualization of radix-sort results.
///
///        Sorts random uint32_t data on the GPU with OpenCL, then renders two
///        point-cloud scatter plots in a GLFW window using Vulkan:
///
///            Top half  — unsorted data
///            Bottom half — sorted data
///
///        Each element is drawn as a coloured point whose X position is its
///        array index and whose Y position is its normalised value.  A heat-map
///        colour scheme (blue → red) encodes the magnitude.
///
/// Dependencies: Vulkan SDK (with glslc), GLFW 3, vk-bootstrap
///
/// Build (from project root):
///   cmake -B build && cmake --build build
///
/// Run:
///   ./build/examples/visualize

// ── Vulkan / windowing ──────────────────────────────────────────────────
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <VkBootstrap.h>

// ── radixsortcl library ─────────────────────────────────────────────────
#include "Common/ComputeState.h"
#include "RadixSortGPU.h"
#include "Dataset.h"
#include "Parameters.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

// =====================================================================
// Constants
// =====================================================================
constexpr uint32_t WINDOW_W          = 1280;
constexpr uint32_t WINDOW_H          = 720;
constexpr uint32_t NUM_ELEMENTS      = 4096;
constexpr int      MAX_FRAMES        = 2;

// =====================================================================
// Push constants – must match the GLSL layout exactly (16 bytes)
// =====================================================================
struct PushConstants {
    uint32_t count;
    float    maxValue;
    float    yOffset;
    float    yScale;
};

// =====================================================================
// Small helpers
// =====================================================================

/// Return the directory that contains the running executable.
static std::filesystem::path exeDir()
{
#ifdef _WIN32
    wchar_t buf[MAX_PATH]{};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path();
#elif defined(__linux__)
    return std::filesystem::canonical("/proc/self/exe").parent_path();
#elif defined(__APPLE__)
    char buf[1024]{};
    uint32_t sz = sizeof(buf);
    if (_NSGetExecutablePath(buf, &sz) == 0)
        return std::filesystem::path(buf).parent_path();
    return {};
#else
    return {};
#endif
}

static std::vector<char> readBinaryFile(const std::string& path)
{
    // Try the given path first, then resolve relative to the executable.
    auto tryOpen = [](const std::filesystem::path& p) -> std::ifstream {
        return std::ifstream(p, std::ios::ate | std::ios::binary);
    };

    std::filesystem::path resolved = path;
    std::ifstream f = tryOpen(resolved);
    if (!f.is_open()) {
        resolved = exeDir() / path;
        f = tryOpen(resolved);
    }
    if (!f.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    const auto size = static_cast<size_t>(f.tellg());
    std::vector<char> buf(size);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

static uint32_t findMemoryType(
    VkPhysicalDevice gpu, uint32_t filter, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        if ((filter & (1u << i)) &&
            (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    throw std::runtime_error("No suitable memory type");
}

// ── Vulkan buffer + device memory pair ──────────────────────────────────
struct GpuBuffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

static GpuBuffer createBuffer(
    VkDevice dev, VkPhysicalDevice gpu,
    VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
{
    GpuBuffer b{};
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size        = size;
    ci.usage       = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(dev, &ci, nullptr, &b.buffer) != VK_SUCCESS)
        throw std::runtime_error("vkCreateBuffer failed");

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(dev, b.buffer, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize  = req.size;
    ai.memoryTypeIndex = findMemoryType(gpu, req.memoryTypeBits, props);
    if (vkAllocateMemory(dev, &ai, nullptr, &b.memory) != VK_SUCCESS)
        throw std::runtime_error("vkAllocateMemory failed");

    vkBindBufferMemory(dev, b.buffer, b.memory, 0);
    return b;
}

static void uploadToBuffer(
    VkDevice dev, const GpuBuffer& buf, const void* src, VkDeviceSize size)
{
    void* dst = nullptr;
    vkMapMemory(dev, buf.memory, 0, size, 0, &dst);
    std::memcpy(dst, src, size);
    vkUnmapMemory(dev, buf.memory);
}

static void destroyBuffer(VkDevice dev, GpuBuffer& b)
{
    if (b.buffer) vkDestroyBuffer(dev, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(dev, b.memory, nullptr);
    b = {};
}

// =====================================================================
// Vulkan application state
// =====================================================================
struct App {
    GLFWwindow*      window = nullptr;

    // vk-bootstrap objects (lifetime-managed)
    vkb::Instance    vkbInst{};
    VkSurfaceKHR     surface  = VK_NULL_HANDLE;
    vkb::Device      vkbDev{};
    vkb::Swapchain   vkbSwap{};

    // Frequently used raw handles
    VkDevice         dev      = VK_NULL_HANDLE;
    VkPhysicalDevice gpu      = VK_NULL_HANDLE;
    VkQueue          gfxQueue = VK_NULL_HANDLE;
    VkQueue          prsQueue = VK_NULL_HANDLE;

    VkFormat         swapFmt{};
    VkExtent2D       swapExt{};
    std::vector<VkImage>     swapImages;
    std::vector<VkImageView> swapViews;

    VkRenderPass         renderPass     = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsLayout      = VK_NULL_HANDLE;
    VkPipelineLayout     pipeLayout     = VK_NULL_HANDLE;
    VkPipeline           pipeline       = VK_NULL_HANDLE;
    VkDescriptorPool     dsPool         = VK_NULL_HANDLE;
    VkCommandPool        cmdPool        = VK_NULL_HANDLE;

    std::vector<VkFramebuffer>   framebuffers;
    std::vector<VkCommandBuffer> cmdBufs;

    // Per-frame sync
    std::vector<VkSemaphore> semImgReady;
    std::vector<VkSemaphore> semRenderDone;
    std::vector<VkFence>     fences;

    // Data buffers + descriptor sets
    GpuBuffer       bufUnsorted{};
    GpuBuffer       bufSorted{};
    VkDescriptorSet dsUnsorted = VK_NULL_HANDLE;
    VkDescriptorSet dsSorted   = VK_NULL_HANDLE;

    uint32_t frame = 0;
};

// ── Initialisation helpers ──────────────────────────────────────────────

static void initWindow(App& a)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);
    a.window = glfwCreateWindow(
        WINDOW_W, WINDOW_H,
        "Radix Sort — Top: unsorted  |  Bottom: sorted",
        nullptr, nullptr);
}

static void initVulkan(App& a)
{
    // Instance
    auto ir = vkb::InstanceBuilder{}
        .set_app_name("RadixSort Visualizer")
        .request_validation_layers()
        .build();
    if (!ir) throw std::runtime_error(ir.error().message());
    a.vkbInst = ir.value();

    // Surface
    if (glfwCreateWindowSurface(a.vkbInst.instance, a.window,
                                nullptr, &a.surface) != VK_SUCCESS)
        throw std::runtime_error("Failed to create window surface");

    // Physical device
    auto pr = vkb::PhysicalDeviceSelector{a.vkbInst}
        .set_surface(a.surface)
        .prefer_gpu_device_type(vkb::PreferredDeviceType::discrete)
        .select();
    if (!pr) throw std::runtime_error(pr.error().message());

    // Logical device
    auto dr = vkb::DeviceBuilder{pr.value()}.build();
    if (!dr) throw std::runtime_error(dr.error().message());
    a.vkbDev = dr.value();
    a.dev    = a.vkbDev.device;
    a.gpu    = pr.value().physical_device;
    a.gfxQueue = a.vkbDev.get_queue(vkb::QueueType::graphics).value();
    a.prsQueue = a.vkbDev.get_queue(vkb::QueueType::present).value();

    // Swapchain
    auto sr = vkb::SwapchainBuilder{a.vkbDev}
        .set_desired_extent(WINDOW_W, WINDOW_H)
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .build();
    if (!sr) throw std::runtime_error(sr.error().message());
    a.vkbSwap   = sr.value();
    a.swapImages = a.vkbSwap.get_images().value();
    a.swapViews  = a.vkbSwap.get_image_views().value();
    a.swapFmt    = a.vkbSwap.image_format;
    a.swapExt    = a.vkbSwap.extent;
}

static void createRenderPass(App& a)
{
    VkAttachmentDescription att{};
    att.format         = a.swapFmt;
    att.samples        = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    att.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};

    VkSubpassDescription sub{};
    sub.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments    = &ref;

    VkSubpassDependency dep{};
    dep.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dep.dstSubpass    = 0;
    dep.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    ci.attachmentCount = 1;  ci.pAttachments  = &att;
    ci.subpassCount    = 1;  ci.pSubpasses    = &sub;
    ci.dependencyCount = 1;  ci.pDependencies = &dep;

    if (vkCreateRenderPass(a.dev, &ci, nullptr, &a.renderPass) != VK_SUCCESS)
        throw std::runtime_error("Failed to create render pass");
}

static VkShaderModule loadShader(VkDevice dev, const std::string& path)
{
    auto code = readBinaryFile(path);
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size();
    ci.pCode    = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule m{};
    if (vkCreateShaderModule(dev, &ci, nullptr, &m) != VK_SUCCESS)
        throw std::runtime_error("Failed to create shader module: " + path);
    return m;
}

static void createDescriptorSetLayout(App& a)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding         = 0;
    b.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo ci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    ci.bindingCount = 1;
    ci.pBindings    = &b;
    if (vkCreateDescriptorSetLayout(a.dev, &ci, nullptr, &a.dsLayout)
        != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor set layout");
}

static void createPipeline(App& a)
{
    VkShaderModule vert = loadShader(a.dev, "visualize.vert.spv");
    VkShaderModule frag = loadShader(a.dev, "visualize.frag.spv");

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage  = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vert;
    stages[0].pName  = "main";
    stages[1].sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = frag;
    stages[1].pName  = "main";

    VkPipelineVertexInputStateCreateInfo vertIn{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};

    VkPipelineInputAssemblyStateCreateInfo ia{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;

    VkViewport vp{0, 0,
        static_cast<float>(a.swapExt.width),
        static_cast<float>(a.swapExt.height), 0, 1};
    VkRect2D sc{{0,0}, a.swapExt};

    VkPipelineViewportStateCreateInfo vs{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1; vs.pViewports = &vp;
    vs.scissorCount  = 1; vs.pScissors  = &sc;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth   = 1.0f;
    rs.cullMode    = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pcr.size       = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 1;
    plci.pSetLayouts            = &a.dsLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(a.dev, &plci, nullptr, &a.pipeLayout)
        != VK_SUCCESS)
        throw std::runtime_error("Failed to create pipeline layout");

    VkGraphicsPipelineCreateInfo pi{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pi.stageCount          = 2;
    pi.pStages             = stages;
    pi.pVertexInputState   = &vertIn;
    pi.pInputAssemblyState = &ia;
    pi.pViewportState      = &vs;
    pi.pRasterizationState = &rs;
    pi.pMultisampleState   = &ms;
    pi.pColorBlendState    = &cb;
    pi.layout              = a.pipeLayout;
    pi.renderPass          = a.renderPass;
    if (vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &pi, nullptr,
                                  &a.pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create graphics pipeline");

    vkDestroyShaderModule(a.dev, vert, nullptr);
    vkDestroyShaderModule(a.dev, frag, nullptr);
}

static void createFramebuffers(App& a)
{
    a.framebuffers.resize(a.swapViews.size());
    for (size_t i = 0; i < a.swapViews.size(); ++i) {
        VkFramebufferCreateInfo ci{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        ci.renderPass      = a.renderPass;
        ci.attachmentCount = 1;
        ci.pAttachments    = &a.swapViews[i];
        ci.width           = a.swapExt.width;
        ci.height          = a.swapExt.height;
        ci.layers          = 1;
        if (vkCreateFramebuffer(a.dev, &ci, nullptr, &a.framebuffers[i])
            != VK_SUCCESS)
            throw std::runtime_error("Failed to create framebuffer");
    }
}

static void createCommandPool(App& a)
{
    auto idx = a.vkbDev.get_queue_index(vkb::QueueType::graphics).value();
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    ci.queueFamilyIndex = idx;
    if (vkCreateCommandPool(a.dev, &ci, nullptr, &a.cmdPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create command pool");
}

static void createDataBuffers(
    App& a,
    const std::vector<uint32_t>& unsorted,
    const std::vector<uint32_t>& sorted)
{
    const VkDeviceSize sz = sizeof(uint32_t) * unsorted.size();
    constexpr auto usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr auto props  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    a.bufUnsorted = createBuffer(a.dev, a.gpu, sz, usage, props);
    uploadToBuffer(a.dev, a.bufUnsorted, unsorted.data(), sz);

    a.bufSorted = createBuffer(a.dev, a.gpu, sz, usage, props);
    uploadToBuffer(a.dev, a.bufSorted, sorted.data(), sz);
}

static void createDescriptorSets(App& a, uint32_t count)
{
    // Pool
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.poolSizeCount = 1;  pci.pPoolSizes = &ps;
    pci.maxSets       = 2;
    if (vkCreateDescriptorPool(a.dev, &pci, nullptr, &a.dsPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");

    // Allocate
    std::array<VkDescriptorSetLayout, 2> layouts = {a.dsLayout, a.dsLayout};
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = a.dsPool;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts        = layouts.data();
    std::array<VkDescriptorSet, 2> sets{};
    if (vkAllocateDescriptorSets(a.dev, &ai, sets.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor sets");
    a.dsUnsorted = sets[0];
    a.dsSorted   = sets[1];

    // Write
    const VkDeviceSize bufSize = sizeof(uint32_t) * count;
    auto writeDS = [&](VkDescriptorSet ds, VkBuffer buf) {
        VkDescriptorBufferInfo bi{buf, 0, bufSize};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = ds;
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
    };
    writeDS(a.dsUnsorted, a.bufUnsorted.buffer);
    writeDS(a.dsSorted,   a.bufSorted.buffer);
}

static void createCommandBuffers(App& a)
{
    a.cmdBufs.resize(MAX_FRAMES);
    VkCommandBufferAllocateInfo ai{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool        = a.cmdPool;
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = MAX_FRAMES;
    if (vkAllocateCommandBuffers(a.dev, &ai, a.cmdBufs.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate command buffers");
}

static void createSyncObjects(App& a)
{
    const auto imageCount = a.swapImages.size();
    a.semImgReady.resize(MAX_FRAMES);
    a.semRenderDone.resize(imageCount);   // one per swapchain image
    a.fences.resize(MAX_FRAMES);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkFenceCreateInfo     fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (int i = 0; i < MAX_FRAMES; ++i) {
        if (vkCreateSemaphore(a.dev, &si, nullptr, &a.semImgReady[i])
            != VK_SUCCESS ||
            vkCreateFence(a.dev, &fi, nullptr, &a.fences[i]) != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects");
    }
    for (size_t i = 0; i < imageCount; ++i) {
        if (vkCreateSemaphore(a.dev, &si, nullptr, &a.semRenderDone[i])
            != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects");
    }
}

// ── Recording & presentation ────────────────────────────────────────────

static void recordFrame(
    App& a, VkCommandBuffer cmd, uint32_t imgIdx,
    uint32_t count, float maxVal)
{
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);

    VkClearValue clear{{{0.04f, 0.04f, 0.08f, 1.0f}}};
    VkRenderPassBeginInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    rp.renderPass        = a.renderPass;
    rp.framebuffer       = a.framebuffers[imgIdx];
    rp.renderArea.extent = a.swapExt;
    rp.clearValueCount   = 1;
    rp.pClearValues      = &clear;

    vkCmdBeginRenderPass(cmd, &rp, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeline);

    // Top half – unsorted
    {
        PushConstants pc{count, maxVal, -0.5f, 0.45f};
        vkCmdPushConstants(cmd, a.pipeLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                a.pipeLayout, 0, 1, &a.dsUnsorted, 0, nullptr);
        vkCmdDraw(cmd, count, 1, 0, 0);
    }
    // Bottom half – sorted
    {
        PushConstants pc{count, maxVal, 0.5f, 0.45f};
        vkCmdPushConstants(cmd, a.pipeLayout,
                           VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                a.pipeLayout, 0, 1, &a.dsSorted, 0, nullptr);
        vkCmdDraw(cmd, count, 1, 0, 0);
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

static void drawFrame(App& a, uint32_t count, float maxVal)
{
    vkWaitForFences(a.dev, 1, &a.fences[a.frame], VK_TRUE, UINT64_MAX);

    uint32_t imgIdx = 0;
    vkAcquireNextImageKHR(a.dev, a.vkbSwap.swapchain, UINT64_MAX,
                          a.semImgReady[a.frame], VK_NULL_HANDLE, &imgIdx);
    vkResetFences(a.dev, 1, &a.fences[a.frame]);

    auto cmd = a.cmdBufs[a.frame];
    vkResetCommandBuffer(cmd, 0);
    recordFrame(a, cmd, imgIdx, count, maxVal);

    VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &a.semImgReady[a.frame];
    si.pWaitDstStageMask    = &wait;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &a.semRenderDone[imgIdx];  // per swapchain image
    if (vkQueueSubmit(a.gfxQueue, 1, &si, a.fences[a.frame]) != VK_SUCCESS)
        throw std::runtime_error("Queue submit failed");

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &a.semRenderDone[imgIdx];  // per swapchain image
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &a.vkbSwap.swapchain;
    pi.pImageIndices      = &imgIdx;
    vkQueuePresentKHR(a.prsQueue, &pi);

    a.frame = (a.frame + 1) % MAX_FRAMES;
}

// ── Cleanup ─────────────────────────────────────────────────────────────

static void cleanup(App& a)
{
    if (a.dev) vkDeviceWaitIdle(a.dev);

    for (int i = 0; i < MAX_FRAMES; ++i) {
        if (a.semImgReady[i]) vkDestroySemaphore(a.dev, a.semImgReady[i], nullptr);
        if (a.fences[i])      vkDestroyFence(a.dev, a.fences[i], nullptr);
    }
    for (auto s : a.semRenderDone)
        if (s) vkDestroySemaphore(a.dev, s, nullptr);
    if (a.cmdPool)    vkDestroyCommandPool(a.dev, a.cmdPool, nullptr);

    destroyBuffer(a.dev, a.bufUnsorted);
    destroyBuffer(a.dev, a.bufSorted);

    if (a.dsPool)     vkDestroyDescriptorPool(a.dev, a.dsPool, nullptr);
    if (a.dsLayout)   vkDestroyDescriptorSetLayout(a.dev, a.dsLayout, nullptr);

    for (auto fb : a.framebuffers) vkDestroyFramebuffer(a.dev, fb, nullptr);

    if (a.pipeline)   vkDestroyPipeline(a.dev, a.pipeline, nullptr);
    if (a.pipeLayout) vkDestroyPipelineLayout(a.dev, a.pipeLayout, nullptr);
    if (a.renderPass) vkDestroyRenderPass(a.dev, a.renderPass, nullptr);

    a.vkbSwap.destroy_image_views(a.swapViews);
    vkb::destroy_swapchain(a.vkbSwap);
    vkb::destroy_device(a.vkbDev);
    if (a.surface) vkDestroySurfaceKHR(a.vkbInst.instance, a.surface, nullptr);
    vkb::destroy_instance(a.vkbInst);

    if (a.window) glfwDestroyWindow(a.window);
    glfwTerminate();
}

// =====================================================================
// OpenCL sorting  (same pattern as the basic_sort example)
// =====================================================================

template <typename DataType>
bool sortData(
    ComputeState& compute, uint32_t numElements,
    std::vector<DataType>& outUnsorted,
    std::vector<DataType>& outSorted)
{
    using Params = AlgorithmParameters<DataType>;

    RandomDistributed<DataType> dataset(numElements);
    RadixSortGPU<DataType> sorter;
    const uint32_t numRounded = sorter.Resize(numElements);

    std::vector<DataType>  hKeys(numRounded);
    std::vector<DataType>  hResult(numRounded);
    std::vector<uint32_t>  hHisto(Params::_RADIX * Params::_NUM_ITEMS);
    std::vector<uint32_t>  hGlobsum(Params::_NUM_HISTOSPLIT);
    std::vector<uint32_t>  hPermut(numRounded);

    std::copy_n(dataset.dataset.begin(), numElements, hKeys.begin());
    std::iota(hPermut.begin(), hPermut.end(), 0U);

    outUnsorted.assign(hKeys.begin(), hKeys.begin() + numElements);

    HostSpans<DataType> spans{
        {hKeys.data(),    hKeys.size()},
        {hHisto.data(),   hHisto.size()},
        {hGlobsum.data(), hGlobsum.size()},
        {hPermut.data(),  hPermut.size()},
        {hResult.data(),  hResult.size()},
    };

    auto status = sorter.initialize(
        compute.device(), compute.m_CLContext,
        numElements, spans);
    if (status != OperationStatus::OK) return false;

    auto& q = compute.m_CLCommandQueue;
    if (numRounded != numElements)
        sorter.padGPUData(q, sizeof(DataType) * numElements);

    status = sorter.uploadData(q);
    if (status != OperationStatus::OK) return false;
    status = sorter.calculate(q);
    if (status != OperationStatus::OK) return false;
    status = sorter.downloadData(q);
    if (status != OperationStatus::OK) return false;

    outSorted.assign(hResult.begin(), hResult.begin() + numElements);
    sorter.release();
    return true;
}

// =====================================================================
// Entry point
// =====================================================================

int main()
{
    // ── 1. Sort with OpenCL ─────────────────────────────────────────
    ComputeState compute;
    try {
        if (!compute.init()) {
            std::cerr << "No suitable OpenCL GPU device found.\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "OpenCL init error: " << e.what() << '\n';
        return 1;
    }

    std::vector<uint32_t> unsorted, sorted;
    std::cout << "Sorting " << NUM_ELEMENTS
              << " uint32_t values on the GPU (OpenCL)...\n";
    if (!sortData<uint32_t>(compute, NUM_ELEMENTS, unsorted, sorted)) {
        std::cerr << "OpenCL sort failed.\n";
        return 1;
    }

    const auto maxVal = static_cast<float>(
        *std::max_element(unsorted.begin(), unsorted.end()));
    std::cout << "Sort complete.  Launching Vulkan visualisation...\n";

    // ── 2. Visualise with Vulkan ────────────────────────────────────
    App app{};
    try {
        initWindow(app);
        initVulkan(app);
        createRenderPass(app);
        createDescriptorSetLayout(app);
        createPipeline(app);
        createFramebuffers(app);
        createCommandPool(app);
        createDataBuffers(app, unsorted, sorted);
        createDescriptorSets(app, NUM_ELEMENTS);
        createCommandBuffers(app);
        createSyncObjects(app);

        while (!glfwWindowShouldClose(app.window)) {
            glfwPollEvents();
            drawFrame(app, NUM_ELEMENTS, maxVal);
        }

        cleanup(app);
    } catch (const std::exception& e) {
        std::cerr << "Vulkan error: " << e.what() << '\n';
        cleanup(app);
        return 1;
    }
    return 0;
}
