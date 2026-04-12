#pragma once

#include "Common/CommonDefs.h"
#include "Common/ComputeState.h"

#include <print>
#include <iostream>
#include <vector>
#include <string>


template<class Task>
concept ComputeTask = requires(Task t, cl::Device dev, cl::Context ctx){
    {t.InitResources(dev, ctx)} -> std::same_as<bool>;
    {t.ReleaseResources()} -> std::same_as<void>;
};

class CTestBase
{
public:
    CTestBase(std::vector<std::string> arguments = {})
        : m_arguments(arguments), m_computeState{}
    { }

	virtual ~CTestBase() = default;

	//! To be overridden
	virtual bool DoCompute() = 0;

	virtual bool InitCLContext() {
        return m_computeState.init();
    }

    bool RunComputeTask(ComputeTask auto& task, const LocalWorkSize& localWorkSize)
    {
        if(m_computeState.m_CLContext() == nullptr)
        {
            std::println(std::cerr, "Error: RunComputeTask() cannot execute because the OpenCL context is null.");
            return false;
        }

        if(!task.InitResources(
                m_computeState.device(),
                m_computeState.m_CLContext
            )
        )
        {
            std::println(std::cerr, "Error during resource allocation. Aborting execution.");
            task.ReleaseResources();
            return false;
        }

        // Compute the golden result.
        std::println("Computing CPU reference result...");
        task.ComputeCPU();
        std::println("DONE");

        // Running the same task on the GPU.
        std::println("Computing GPU result...");

        // Runing the kernel N times. This make the measurement of the execution time more accurate.
        task.ComputeGPU(
                m_computeState.m_CLContext,
                m_computeState.m_CLCommandQueue,
                localWorkSize
        );
        std::println("DONE");

        // Validating results.
        std::string result = "GOLD TEST PASSED!\n";
        if (!task.ValidateResults())
        {
            result = "INVALID RESULTS!\n";
        }
        std::println("{}", task.ValidateResults() ? "GOLD TEST PASSED!" : "INVALID RESULTS");

        // Cleaning up.
        task.ReleaseResources();

        return true;
    }

protected:
    ComputeState m_computeState;

    std::vector<std::string> m_arguments;
};

