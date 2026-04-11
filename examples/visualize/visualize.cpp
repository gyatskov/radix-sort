/// @file visualize.cpp
/// @brief Vulkan visualization of radix-sort results.
///
///        Sorts random uint32_t data on the GPU with OpenCL, then renders
///        multiple point-cloud scatter plots in a GLFW window using Vulkan:
///
///            Row 0       — unsorted input data
///            Rows 1..N   — intermediate state after each radix sort pass
///            Row N       — fully sorted data (final pass)
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
#include <cassert>
#include <chrono>
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
constexpr uint32_t NUM_PASSES        = AlgorithmParameters<uint32_t>::_NUM_PASSES;
constexpr uint32_t TOTAL_ROWS        = NUM_PASSES + 1;  // unsorted + one per pass
constexpr int      MAX_FRAMES        = 2;

static bool g_regenerate = false;
static bool g_framebufferResized = false;

// =====================================================================
// Push constants – must match the GLSL layout exactly
// =====================================================================
struct PushConstants {
    uint32_t count;
    float    maxValue;
    float    yOffset;
    float    yScale;
    float    xOffset;   // horizontal center of column in NDC
    float    xScale;    // horizontal half-width (fraction of full)
};

struct OverlayPushConstants {
    float    value;      // display value (timeMs or pass index)
    float    anchorX;    // NDC X of left edge of text block
    float    anchorY;    // NDC Y of top edge of text block
    float    charW;
    float    charH;
    uint32_t numChars;
    uint32_t mode;       // 0 = time ("NNN.NN ms"), 1 = integer digits
};

constexpr uint32_t OVERLAY_NUM_CHARS = 9;  // "NNN.NN ms"

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

    // Overlay (time display)
    VkPipelineLayout     overlayPipeLayout = VK_NULL_HANDLE;
    VkPipeline           overlayPipeline   = VK_NULL_HANDLE;

    std::vector<VkFramebuffer>   framebuffers;
    std::vector<VkCommandBuffer> cmdBufs;

    // Per-frame sync
    std::vector<VkSemaphore> semImgReady;
    std::vector<VkSemaphore> semRenderDone;
    std::vector<VkFence>     fences;

    // Data buffers + descriptor sets (persistently mapped for zero-copy)
    GpuBuffer       bufUnsorted{};
    void*           mappedUnsorted = nullptr;
    VkDescriptorSet dsUnsorted = VK_NULL_HANDLE;

    // Column types: keys, histograms, globsum, inputPermut, outputPermut
    static constexpr uint32_t NUM_COLS = 5;
    struct ColData {
        std::vector<GpuBuffer>       bufs;      // [NUM_PASSES]
        std::vector<void*>           mapped;    // [NUM_PASSES]
        std::vector<VkDescriptorSet> ds;        // [NUM_PASSES]
        uint32_t elemCount = 0;
        float    maxVal    = 1.0f;
    };
    ColData cols[NUM_COLS];  // 0=keys, 1=histo, 2=globsum, 3=inPermut, 4=outPermut

    uint32_t frame = 0;
};

// ── Initialisation helpers ──────────────────────────────────────────────

static void initWindow(App& a)
{
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_TRUE);
    a.window = glfwCreateWindow(
        WINDOW_W, WINDOW_H,
        "Radix Sort \xe2\x80\x94 Unsorted + Pass Intermediates  (click to regenerate)",
        nullptr, nullptr);
    glfwSetMouseButtonCallback(a.window,
        [](GLFWwindow*, int button, int action, int) {
            if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS)
                g_regenerate = true;
        });
    glfwSetFramebufferSizeCallback(a.window,
        [](GLFWwindow*, int, int) {
            g_framebufferResized = true;
        });
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

    VkPipelineViewportStateCreateInfo vs{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1;
    vs.scissorCount  = 1;

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

    VkDynamicState dynStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

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
    pi.pDynamicState       = &dyn;
    pi.layout              = a.pipeLayout;
    pi.renderPass          = a.renderPass;
    if (vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &pi, nullptr,
                                  &a.pipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create graphics pipeline");

    vkDestroyShaderModule(a.dev, vert, nullptr);
    vkDestroyShaderModule(a.dev, frag, nullptr);
}

static void createOverlayPipeline(App& a)
{
    VkShaderModule vert = loadShader(a.dev, "overlay.vert.spv");
    VkShaderModule frag = loadShader(a.dev, "overlay.frag.spv");

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
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vs{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = 1;
    vs.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rs{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth   = 1.0f;
    rs.cullMode    = VK_CULL_MODE_NONE;

    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState cba{};
    cba.blendEnable         = VK_TRUE;
    cba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    cba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    cba.colorBlendOp        = VK_BLEND_OP_ADD;
    cba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    cba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    cba.alphaBlendOp        = VK_BLEND_OP_ADD;
    cba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                         VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = 1;
    cb.pAttachments    = &cba;

    VkDynamicState dynStates[] = {
        VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates    = dynStates;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.size       = sizeof(OverlayPushConstants);

    VkPipelineLayoutCreateInfo plci{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pcr;
    if (vkCreatePipelineLayout(a.dev, &plci, nullptr, &a.overlayPipeLayout)
        != VK_SUCCESS)
        throw std::runtime_error("Failed to create overlay pipeline layout");

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
    pi.pDynamicState       = &dyn;
    pi.layout              = a.overlayPipeLayout;
    pi.renderPass          = a.renderPass;
    if (vkCreateGraphicsPipelines(a.dev, VK_NULL_HANDLE, 1, &pi, nullptr,
                                  &a.overlayPipeline) != VK_SUCCESS)
        throw std::runtime_error("Failed to create overlay pipeline");

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

/// Create Vulkan storage buffers and persistently map them so that OpenCL
/// DMA transfers can read/write directly — no intermediate host copies.
static void createMappedDataBuffers(App& a, uint32_t numRounded)
{
    using Params = AlgorithmParameters<uint32_t>;
    constexpr auto usage  = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr auto props  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    // Unsorted keys buffer
    const VkDeviceSize szKeys = sizeof(uint32_t) * numRounded;
    a.bufUnsorted = createBuffer(a.dev, a.gpu, szKeys, usage, props);
    vkMapMemory(a.dev, a.bufUnsorted.memory, 0, szKeys, 0, &a.mappedUnsorted);

    // Per-column element counts and buffer sizes
    const uint32_t colCounts[App::NUM_COLS] = {
        numRounded,                                    // keys
        AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS,           // histograms
        AlgorithmConfiguration::_NUM_HISTOSPLIT,                       // globsum
        numRounded,                                    // input permutations
        numRounded,                                    // output permutations
    };

    for (uint32_t c = 0; c < App::NUM_COLS; ++c) {
        auto& col = a.cols[c];
        col.elemCount = colCounts[c];
        col.bufs.resize(NUM_PASSES);
        col.mapped.resize(NUM_PASSES, nullptr);
        const VkDeviceSize sz = sizeof(uint32_t) * col.elemCount;
        for (uint32_t p = 0; p < NUM_PASSES; ++p) {
            col.bufs[p] = createBuffer(a.dev, a.gpu, sz, usage, props);
            vkMapMemory(a.dev, col.bufs[p].memory, 0, sz, 0, &col.mapped[p]);
        }
    }
}

static void createDescriptorSets(App& a)
{
    // 1 for unsorted keys + NUM_COLS * NUM_PASSES for the per-column grids
    constexpr uint32_t totalSets = 1 + App::NUM_COLS * NUM_PASSES;

    // Pool
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, totalSets};
    VkDescriptorPoolCreateInfo pci{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.poolSizeCount = 1;  pci.pPoolSizes = &ps;
    pci.maxSets       = totalSets;
    if (vkCreateDescriptorPool(a.dev, &pci, nullptr, &a.dsPool) != VK_SUCCESS)
        throw std::runtime_error("Failed to create descriptor pool");

    // Allocate all at once
    std::vector<VkDescriptorSetLayout> layouts(totalSets, a.dsLayout);
    VkDescriptorSetAllocateInfo ai{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool     = a.dsPool;
    ai.descriptorSetCount = totalSets;
    ai.pSetLayouts        = layouts.data();
    std::vector<VkDescriptorSet> sets(totalSets);
    if (vkAllocateDescriptorSets(a.dev, &ai, sets.data()) != VK_SUCCESS)
        throw std::runtime_error("Failed to allocate descriptor sets");

    // Distribute descriptor sets
    a.dsUnsorted = sets[0];
    uint32_t idx = 1;
    for (uint32_t c = 0; c < App::NUM_COLS; ++c) {
        a.cols[c].ds.resize(NUM_PASSES);
        for (uint32_t p = 0; p < NUM_PASSES; ++p)
            a.cols[c].ds[p] = sets[idx++];
    }

    // Write descriptor bindings
    auto writeDS = [&](VkDescriptorSet ds, VkBuffer buf, VkDeviceSize bufSize) {
        VkDescriptorBufferInfo bi{buf, 0, bufSize};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet          = ds;
        w.dstBinding      = 0;
        w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.descriptorCount = 1;
        w.pBufferInfo     = &bi;
        vkUpdateDescriptorSets(a.dev, 1, &w, 0, nullptr);
    };
    writeDS(a.dsUnsorted, a.bufUnsorted.buffer,
            sizeof(uint32_t) * a.cols[0].elemCount);
    for (uint32_t c = 0; c < App::NUM_COLS; ++c) {
        const VkDeviceSize sz = sizeof(uint32_t) * a.cols[c].elemCount;
        for (uint32_t p = 0; p < NUM_PASSES; ++p)
            writeDS(a.cols[c].ds[p], a.cols[c].bufs[p].buffer, sz);
    }
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
    App& a, VkCommandBuffer cmd, uint32_t imgIdx, float timeMs)
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

    VkViewport vp{0, 0,
        static_cast<float>(a.swapExt.width),
        static_cast<float>(a.swapExt.height), 0, 1};
    VkRect2D sc{{0,0}, a.swapExt};
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &sc);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, a.pipeline);

    // Grid layout: NUM_COLS columns × TOTAL_ROWS rows.
    // Row 0 = unsorted keys (only in column 0).
    // Rows 1..NUM_PASSES = per-pass state for all columns.
    constexpr float colWidth = 1.0f / App::NUM_COLS;  // in half-NDC

    for (uint32_t col = 0; col < App::NUM_COLS; ++col) {
        const auto& cd = a.cols[col];
        const float xCenter = -1.0f + (2.0f * col + 1.0f) / App::NUM_COLS;

        for (uint32_t row = 0; row < TOTAL_ROWS; ++row) {
            const float yOffset = -1.0f + 2.0f * (row + 0.5f) / TOTAL_ROWS;
            const float yScale  = 0.9f / TOTAL_ROWS;

            // Row 0 is unsorted keys — only column 0 draws it
            VkDescriptorSet ds;
            uint32_t count;
            float maxVal;
            if (row == 0) {
                if (col != 0) continue;
                ds     = a.dsUnsorted;
                count  = cd.elemCount;
                maxVal = cd.maxVal;
            } else {
                ds     = cd.ds[row - 1];
                count  = cd.elemCount;
                maxVal = cd.maxVal;
            }

            PushConstants pc{};
            pc.count    = count;
            pc.maxValue = maxVal;
            pc.yOffset  = yOffset;
            pc.yScale   = yScale;
            if (row == 0) {
                // Unsorted keys span full width
                pc.xOffset = 0.0f;
                pc.xScale  = 1.0f;
            } else {
                pc.xOffset = xCenter;
                pc.xScale  = colWidth * 0.95f;
            }
            vkCmdPushConstants(cmd, a.pipeLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pc), &pc);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    a.pipeLayout, 0, 1, &ds, 0, nullptr);
            vkCmdDraw(cmd, count, 1, 0, 0);
        }
    }

    // Overlay: sort time in bottom-right
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, a.overlayPipeline);
    {
        constexpr float charW = 0.025f;
        constexpr float charH = 0.06f;
        constexpr float gap   = charW * 0.3f;
        constexpr float stride = charW + gap;

        // Timer label — bottom-right corner
        {
            constexpr float totalW = OVERLAY_NUM_CHARS * stride - gap;
            OverlayPushConstants opc{};
            opc.value    = timeMs;
            opc.anchorX  = 1.0f - 0.02f - totalW;
            opc.anchorY  = 1.0f - 0.02f - charH;
            opc.charW    = charW;
            opc.charH    = charH;
            opc.numChars = OVERLAY_NUM_CHARS;
            opc.mode     = 0;
            vkCmdPushConstants(cmd, a.overlayPipeLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(opc), &opc);
            vkCmdDraw(cmd, OVERLAY_NUM_CHARS * 6, 1, 0, 0);
        }

        // Pass index labels — left side of each pass row
        for (uint32_t row = 1; row < TOTAL_ROWS; ++row) {
            const float yCenter = -1.0f + 2.0f * (row + 0.5f) / TOTAL_ROWS;
            OverlayPushConstants opc{};
            opc.value    = static_cast<float>(row - 1);
            opc.anchorX  = -1.0f + 0.01f;
            opc.anchorY  = yCenter - charH * 0.5f;
            opc.charW    = charW;
            opc.charH    = charH;
            opc.numChars = 1;
            opc.mode     = 1;
            vkCmdPushConstants(cmd, a.overlayPipeLayout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(opc), &opc);
            vkCmdDraw(cmd, 6, 1, 0, 0);
        }

        // Column header labels — top of each column (row 1)
        {
            // Pack 4 ASCII chars into a uint32 (byte 0 = first char)
            auto packChars = [](const char s[4]) -> uint32_t {
                return uint32_t(s[0])
                     | (uint32_t(s[1]) << 8)
                     | (uint32_t(s[2]) << 16)
                     | (uint32_t(s[3]) << 24);
            };
            const uint32_t colLabels[App::NUM_COLS] = {
                packChars("KEYS"),
                packChars("HIST"),
                packChars("GSUM"),
                packChars("IPMT"),
                packChars("OPMT"),
            };
            // Place labels at the top of the first pass row (row 1)
            const float rowY = -1.0f + 2.0f * (1.0f + 0.5f) / TOTAL_ROWS;
            constexpr uint32_t labelLen = 4;
            const float labelW = labelLen * stride - gap;
            for (uint32_t col = 0; col < App::NUM_COLS; ++col) {
                const float xCenter = -1.0f + (2.0f * col + 1.0f) / App::NUM_COLS;
                OverlayPushConstants opc{};
                float packed;
                std::memcpy(&packed, &colLabels[col], sizeof(float));
                opc.value    = packed;
                opc.anchorX  = xCenter - labelW * 0.5f;
                opc.anchorY  = rowY - 0.9f / TOTAL_ROWS - charH;
                opc.charW    = charW;
                opc.charH    = charH;
                opc.numChars = labelLen;
                opc.mode     = 2;
                vkCmdPushConstants(cmd, a.overlayPipeLayout,
                    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                    0, sizeof(opc), &opc);
                vkCmdDraw(cmd, labelLen * 6, 1, 0, 0);
            }
        }
    }

    vkCmdEndRenderPass(cmd);
    vkEndCommandBuffer(cmd);
}

static void recreateSwapchain(App& a);

static void drawFrame(App& a, float timeMs)
{
    vkWaitForFences(a.dev, 1, &a.fences[a.frame], VK_TRUE, UINT64_MAX);

    uint32_t imgIdx = 0;
    VkResult acqResult = vkAcquireNextImageKHR(
        a.dev, a.vkbSwap.swapchain, UINT64_MAX,
        a.semImgReady[a.frame], VK_NULL_HANDLE, &imgIdx);
    if (acqResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain(a);
        return;
    }
    if (acqResult != VK_SUCCESS && acqResult != VK_SUBOPTIMAL_KHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    vkResetFences(a.dev, 1, &a.fences[a.frame]);

    auto cmd = a.cmdBufs[a.frame];
    vkResetCommandBuffer(cmd, 0);
    recordFrame(a, cmd, imgIdx, timeMs);

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
    VkResult prsResult = vkQueuePresentKHR(a.prsQueue, &pi);
    if (prsResult == VK_ERROR_OUT_OF_DATE_KHR ||
        prsResult == VK_SUBOPTIMAL_KHR || g_framebufferResized) {
        g_framebufferResized = false;
        recreateSwapchain(a);
    }

    a.frame = (a.frame + 1) % MAX_FRAMES;
}

// ── Swapchain recreation ─────────────────────────────────────────────────

static void recreateSwapchain(App& a)
{
    int w = 0, h = 0;
    glfwGetFramebufferSize(a.window, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(a.window, &w, &h);
        glfwWaitEvents();
    }
    vkDeviceWaitIdle(a.dev);

    // Destroy old framebuffers
    for (auto fb : a.framebuffers) vkDestroyFramebuffer(a.dev, fb, nullptr);
    a.framebuffers.clear();

    // Destroy old render-done semaphores (one per swapchain image)
    for (auto s : a.semRenderDone)
        if (s) vkDestroySemaphore(a.dev, s, nullptr);
    a.semRenderDone.clear();

    // Destroy old image views and swapchain
    a.vkbSwap.destroy_image_views(a.swapViews);

    // Rebuild swapchain reusing the old one
    auto sr = vkb::SwapchainBuilder{a.vkbDev}
        .set_old_swapchain(a.vkbSwap)
        .set_desired_present_mode(VK_PRESENT_MODE_FIFO_KHR)
        .build();
    if (!sr) throw std::runtime_error(sr.error().message());
    vkb::destroy_swapchain(a.vkbSwap);
    a.vkbSwap   = sr.value();
    a.swapImages = a.vkbSwap.get_images().value();
    a.swapViews  = a.vkbSwap.get_image_views().value();
    a.swapFmt    = a.vkbSwap.image_format;
    a.swapExt    = a.vkbSwap.extent;

    // Recreate framebuffers and per-image semaphores
    createFramebuffers(a);

    const auto imageCount = a.swapImages.size();
    a.semRenderDone.resize(imageCount);
    VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (size_t i = 0; i < imageCount; ++i) {
        if (vkCreateSemaphore(a.dev, &si, nullptr, &a.semRenderDone[i])
            != VK_SUCCESS)
            throw std::runtime_error("Failed to create sync objects");
    }
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

    if (a.mappedUnsorted) { vkUnmapMemory(a.dev, a.bufUnsorted.memory); a.mappedUnsorted = nullptr; }
    for (auto& col : a.cols) {
        for (size_t i = 0; i < col.mapped.size(); ++i) {
            if (col.mapped[i]) {
                vkUnmapMemory(a.dev, col.bufs[i].memory);
                col.mapped[i] = nullptr;
            }
        }
    }
    destroyBuffer(a.dev, a.bufUnsorted);
    for (auto& col : a.cols)
        for (auto& buf : col.bufs)
            destroyBuffer(a.dev, buf);

    if (a.dsPool)     vkDestroyDescriptorPool(a.dev, a.dsPool, nullptr);
    if (a.dsLayout)   vkDestroyDescriptorSetLayout(a.dev, a.dsLayout, nullptr);

    for (auto fb : a.framebuffers) vkDestroyFramebuffer(a.dev, fb, nullptr);

    if (a.overlayPipeline)   vkDestroyPipeline(a.dev, a.overlayPipeline, nullptr);
    if (a.overlayPipeLayout) vkDestroyPipelineLayout(a.dev, a.overlayPipeLayout, nullptr);
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

/// Zero-copy sort: OpenCL reads input from / writes output to the
/// persistently-mapped Vulkan buffers via DMA — no intermediate vectors.
/// Each radix sort pass result is downloaded into per-column Vulkan buffers
/// so the visualizer can display intermediate states for all buffer types.
template <typename DataType>
bool sortDataZeroCopy(
    ComputeState& compute, uint32_t numElements,
    DataType* dstUnsorted,   // persistently mapped Vulkan unsorted buffer
    App& app,                // columns + mapped Vulkan buffers
    uint32_t numRounded)
{
    using Params = AlgorithmParameters<DataType>;
    constexpr auto numPasses = Params::_NUM_PASSES;

    RandomDistributed<DataType> dataset(numElements);
    RadixSortGPU<DataType> sorter;
    [[maybe_unused]] const uint32_t nr = RadixSortGPU<DataType>::Resize(numElements);
    assert(nr == numRounded);

    // Write random data directly into the mapped Vulkan unsorted buffer.
    std::copy_n(dataset.dataset.begin(), numElements, dstUnsorted);
    std::fill_n(dstUnsorted + numElements, numRounded - numElements, DataType{});

    // Last pass buffer for keys doubles as the sorted output destination.
    auto* dstSorted = static_cast<DataType*>(app.cols[0].mapped.back());

    // Scratch buffers for auxiliary data (downloaded then copied to Vulkan).
    std::vector<uint32_t> hHisto(AlgorithmConfiguration::_RADIX * AlgorithmConfiguration::_NUM_ITEMS);
    std::vector<uint32_t> hGlobsum(AlgorithmConfiguration::_NUM_HISTOSPLIT);
    std::vector<uint32_t> hPermut(numRounded);
    std::vector<uint32_t> hOutPermut(numRounded);
    std::iota(hPermut.begin(), hPermut.end(), 0U);

    HostSpans<DataType> spans{
        {dstUnsorted,       numRounded},
        {hHisto.data(),     hHisto.size()},
        {hGlobsum.data(),   hGlobsum.size()},
        {hPermut.data(),    hPermut.size()},
        {hOutPermut.data(), hOutPermut.size()},
        {dstSorted,         numRounded},
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

    // Run pass-by-pass, capturing all intermediate buffer states.
    for (uint32_t pass = 0; pass < numPasses; ++pass) {
        sorter.Histogram(q, pass);
        sorter.ScanHistogram(q);
        sorter.Reorder(q, pass);

        // Download all buffers to scratch
        status = sorter.downloadKeys(q);
        if (status != OperationStatus::OK) return false;
        status = sorter.downloadIntermediate(q);
        if (status != OperationStatus::OK) return false;

        // Copy each buffer type to its per-pass Vulkan buffer.
        // Col 0 = keys: for last pass, dstSorted already points there.
        if (pass < numPasses - 1) {
            std::memcpy(app.cols[0].mapped[pass], dstSorted,
                        sizeof(DataType) * numRounded);
        }
        // Col 1 = histograms
        std::memcpy(app.cols[1].mapped[pass], hHisto.data(),
                    sizeof(uint32_t) * hHisto.size());
        // Col 2 = globsum
        std::memcpy(app.cols[2].mapped[pass], hGlobsum.data(),
                    sizeof(uint32_t) * hGlobsum.size());
        // Col 3 = input permutations
        std::memcpy(app.cols[3].mapped[pass], hPermut.data(),
                    sizeof(uint32_t) * numRounded);
        // Col 4 = output permutations
        std::memcpy(app.cols[4].mapped[pass], hOutPermut.data(),
                    sizeof(uint32_t) * numRounded);
    }

    // Compute max values for each column (for normalization)
    app.cols[0].maxVal = static_cast<float>(
        *std::max_element(dstUnsorted, dstUnsorted + numElements));
    for (uint32_t c = 1; c < App::NUM_COLS; ++c) {
        float maxV = 1.0f;
        for (uint32_t p = 0; p < numPasses; ++p) {
            auto* data = static_cast<uint32_t*>(app.cols[c].mapped[p]);
            auto m = *std::max_element(data, data + app.cols[c].elemCount);
            if (static_cast<float>(m) > maxV) maxV = static_cast<float>(m);
        }
        app.cols[c].maxVal = maxV;
    }

    sorter.release();
    return true;
}

// =====================================================================
// Entry point
// =====================================================================

int main()
{
    // ── 1. Init OpenCL ──────────────────────────────────────────────
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

    // ── 2. Init Vulkan + create persistently-mapped buffers ─────────
    App app{};
    try {
        initWindow(app);
        initVulkan(app);
        createRenderPass(app);
        createDescriptorSetLayout(app);
        createPipeline(app);
        createOverlayPipeline(app);
        createFramebuffers(app);
        createCommandPool(app);

        // Determine padded size and create mapped Vulkan storage buffers.
        const uint32_t numRounded = RadixSortGPU<uint32_t>{}.Resize(NUM_ELEMENTS);
        createMappedDataBuffers(app, numRounded);
        auto* unsortedPtr = static_cast<uint32_t*>(app.mappedUnsorted);

        // ── 3. Sort directly into mapped Vulkan buffers (zero-copy) ─
        std::cout << "Sorting " << NUM_ELEMENTS
                  << " uint32_t values on the GPU (OpenCL)...\n";
        auto t0 = std::chrono::steady_clock::now();
        if (!sortDataZeroCopy<uint32_t>(compute, NUM_ELEMENTS,
                                         unsortedPtr, app, numRounded)) {
            std::cerr << "OpenCL sort failed.\n";
            cleanup(app);
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        float sortTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();

        std::cout << "Sort complete.  Launching Vulkan visualisation...\n";

        createDescriptorSets(app);
        createCommandBuffers(app);
        createSyncObjects(app);

        while (!glfwWindowShouldClose(app.window)) {
            glfwPollEvents();
            if (g_regenerate) {
                g_regenerate = false;
                vkDeviceWaitIdle(app.dev);
                t0 = std::chrono::steady_clock::now();
                if (sortDataZeroCopy<uint32_t>(compute, NUM_ELEMENTS,
                                                unsortedPtr, app, numRounded)) {
                    t1 = std::chrono::steady_clock::now();
                    sortTimeMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
                }
            }
            drawFrame(app, sortTimeMs);
        }

        cleanup(app);
    } catch (const std::exception& e) {
        std::cerr << "Vulkan error: " << e.what() << '\n';
        cleanup(app);
        return 1;
    }
    return 0;
}
