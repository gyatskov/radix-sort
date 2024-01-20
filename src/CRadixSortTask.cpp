#include "CRadixSortTask.h"
#include "CRadixSortCPU.h"
#include "RadixSortOptions.h"

#include "../Common/CLUtil.h"
#include "../Common/CTimer.h"
#include "../Common/CLTypeInformation.h"

#include "Dataset.h"

#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <ctime>        // std::time_t, struct std::tm, std::localtime
#include <chrono>       // std::chrono::system_clock
#include <functional>
#include <type_traits>
#include <cstring>      // memcmp
#include <cassert>

#include <sys/stat.h>

//#define MORE_PROFILING

///
/// Sorts data on CPU using algorithm provided by stdlib
/// @tparam DataType Type of data to be sorted
///
template<typename DataType>
void SortDataSTL(
    const CheapSpan<DataType>& input,
    CheapSpan<DataType>& output)
{
    std::copy(
        input.data,
        input.data + input.length,
        output.data
    );

    // Inplace reference sorting (STL quicksort):
    std::sort(
        output.data,
        output.data + output.length
    );
}

///
/// Sorts data on CPU using Radix Sort
/// @tparam DataType Type of data to be sorted
///
template<typename DataType>
void SortDataRadix(
    const CheapSpan<DataType>& input,
    CheapSpan<DataType>& output)
{
    std::copy(
        input.data,
        input.data + input.length,
        output.data
    );

    // Reference sorting implementation on CPU (radixsort):
    RadixSortCPU<DataType>::sort(output);
}

template <typename DataType>
CRadixSortTask<DataType>::CRadixSortTask(
    const RadixSortOptions& options,
    std::shared_ptr<Dataset<DataType>> dataset
)
	:
    mNumberKeys(static_cast<decltype(mNumberKeys)>(options.num_elements)),
    // TODO: Check value for initialization
	mNumberKeysRounded(Parameters::_NUM_MAX_INPUT_ELEMS),
	mHostData(dataset),
    m_selectedDataset(dataset),
    mOptions(options)
{}

template <typename DataType>
CRadixSortTask<DataType>::~CRadixSortTask()
{
	ReleaseResources();
}

template <typename DataType>
bool CRadixSortTask<DataType>::InitResources(
    cl_device_id Device,
    cl_context Context)
{
    // CPU resources

    //for (size_t i = 0; i < Parameters::_NUM_MAX_INPUT_ELEMS; i++) {
    //	//m_hInput[i] = m_N - i;			// Use this for debugging
    //	// Mersienne twister
    //	m_hKeys[i] = dis(generator);
    //	//m_hInput[i] = rand() & 15;
    //}

    //std::copy(m_hInput.begin(),
    //	m_hInput.begin() + 100,
    //	std::ostream_iterator<DataType>(std::cout, "\n"));

    CheckLocalMemory(Device);
	mNumberKeysRounded = Resize(mNumberKeys);
    auto& hostBuffers {mHostData.mHostBuffers};
    hostBuffers.m_hResultFromGPU.resize(
        mNumberKeysRounded
    );

    // Collect pointers to host memory
    HostSpans<DataType> hostSpans {
        {hostBuffers.m_hKeys.data(), hostBuffers.m_hKeys.size()},
        {hostBuffers.m_hHistograms.data(), hostBuffers.m_hKeys.size()},
        {hostBuffers.m_hGlobsum.data(), hostBuffers.m_hKeys.size()},
        {hostBuffers.h_Permut.data(), hostBuffers.m_hKeys.size()},
        {hostBuffers.m_hResultFromGPU.data(), hostBuffers.m_hKeys.size()},
    };
    // Initialize actual GPU algorithms and memory
    const auto status = mRadixSortGPU.initialize(
        Device,
        Context,
        mNumberKeys,
        hostSpans
    );

	return status == OperationStatus::OK;
}

template <typename DataType>
void CRadixSortTask<DataType>::ReleaseResources()
{
	// free device resources
    // implicitly done by destructor of ComputeDeviceData
}

template <typename DataType>
void CRadixSortTask<DataType>::ComputeGPU(
    cl_context Context,
    cl_command_queue CommandQueue,
    const std::array<size_t,3>& LocalWorkSize)
{
	if (const auto paddingRequired = mNumberKeys != mNumberKeysRounded) {
        assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
        const auto paddingOffset = sizeof(DataType) * mNumberKeys;
        mRadixSortGPU.padGPUData(CommandQueue, paddingOffset);
    }
	ExecuteTask(Context, CommandQueue, LocalWorkSize);

    // TODO: Extract
    {
        //finish all before we start measuring the time
        V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");
        if (mOptions.perf_to_stdout) {
            const auto& timesCPU {mRuntimesCPU};
            std::cout << " radixsort cpu avg time: "
                      << timesCPU.timeRadix.avg
                      << " ms, throughput: "
                      << 1.0e-6 * (double)mNumberKeysRounded / timesCPU.timeRadix.avg << " Gelem/s"
                      << std::endl;

            std::cout << " stl cpu avg time: "
                      << timesCPU.timeSTL.avg
                      << " ms, throughput: "
                      << 1.0e-6 * (double)mNumberKeysRounded / timesCPU.timeSTL.avg  << " Gelem/s"
                      << std::endl;

            std::cout << "Testing performance of GPU task "
                << "RadixSort" << std::endl;
        }

        TestPerformance(
            CommandQueue,
            [&]() {
                ExecuteTask(Context, CommandQueue, LocalWorkSize);

                return std::make_pair(
                    mRadixSortGPU.getRuntimes(),
                    mRuntimesCPU);
            },
            mOptions,
            Parameters::_NUM_PERFORMANCE_ITERATIONS,
            mNumberKeysRounded,
            m_selectedDataset->name(),
            TypeNameString<DataType>::stdint_name
        );
    }
}

template <typename DataType>
void CRadixSortTask<DataType>::ComputeCPU()
{
    auto& hostBuffers {mHostData.mHostBuffers};
    hostBuffers.m_hKeys.resize(mNumberKeysRounded);

    CheapSpan<DataType> dataInput {
        hostBuffers.m_hKeys.data(),
        hostBuffers.m_hKeys.size(),
    };
    // compute STL result
    {
        CheapSpan<DataType> dataOutput {
            mHostData.m_resultSTLCPU.data(),
            mHostData.m_resultSTLCPU.size(),
        };
        mHostData.m_resultSTLCPU.resize(mNumberKeysRounded);
        CTimer timer;
        timer.Start();
        for (auto j = 0U; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
            SortDataSTL(
                dataInput,
                dataOutput
            );
        }
        timer.Stop();
        mRuntimesCPU.timeSTL.avg =
            timer.GetElapsedMilliseconds() / Parameters::_NUM_PERFORMANCE_ITERATIONS;

    }

    // compute CPU Radix Sort result
    {
        CheapSpan<DataType> dataOutput {
            mHostData.m_resultRadixSortCPU.data(),
            mHostData.m_resultRadixSortCPU.size(),
        };
        mHostData.m_resultRadixSortCPU.resize(mNumberKeysRounded);
        CTimer timer;
        timer.Start();
        for (auto j = 0U; j < Parameters::_NUM_PERFORMANCE_ITERATIONS; j++) {
            SortDataRadix(
                dataInput,
                dataOutput
            );
        }
        timer.Stop();
        mRuntimesCPU.timeRadix.avg =
            timer.GetElapsedMilliseconds() / Parameters::_NUM_PERFORMANCE_ITERATIONS;
    }
}

template <typename DataType>
bool CRadixSortTask<DataType>::ValidateResults()
{
	bool success = true;

    const bool sortedCPU =
        memcmp(
            mHostData.m_resultRadixSortCPU.data(),
            mHostData.m_resultSTLCPU.data(),
            sizeof(DataType) * mNumberKeys) == 0;
    const std::string hasPassedCPU = sortedCPU ? "passed" : "FAILED";

    std::cout << "Data set: " << m_selectedDataset->name() << std::endl;
    std::cout << "Data type: " << TypeNameString<DataType>::stdint_name << std::endl;
    std::cout << "Validation of CPU RadixSort has " + hasPassedCPU << std::endl;
    success = success && sortedCPU;
    const bool sortedGPU =
        std::memcmp(
            mHostData.mHostBuffers.m_hResultFromGPU.data(),
            mHostData.m_resultSTLCPU.data(),
            sizeof(DataType) * mNumberKeys) == 0;

    const std::string hasPassedGPU = sortedGPU ? "passed" : "FAILED";

    std::cout << "Validation of GPU RadixSort has " + hasPassedGPU << std::endl;
    success = success && sortedGPU;

	return success;
}

template <typename DataType>
void CRadixSortTask<DataType>::CheckLocalMemory(cl_device_id Device)
{
    // check that the local mem is sufficient (suggestion of Jose Luis Cerc\F3s Pita)
    cl_ulong localMem{0};
	clGetDeviceInfo(Device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    if (mOptions.verbose) {
        std::cout << "Cache size   = " << localMem << " Bytes." << std::endl;
		std::cout << "Needed cache = " << sizeof(cl_uint) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP << " Bytes." << std::endl;
    }
	assert(localMem > sizeof(DataType) * Parameters::_RADIX * Parameters::_NUM_ITEMS_PER_GROUP);

	constexpr uint32_t maxmemcache =
        std::max(
            Parameters::_NUM_HISTOSPLIT,
		    Parameters::_NUM_ITEMS_PER_GROUP * Parameters::_NUM_GROUPS * Parameters::_RADIX / Parameters::_NUM_HISTOSPLIT
        );
	assert(localMem > sizeof(DataType)*maxmemcache);
}

/// resize the sorted vector
template <typename DataType>
uint32_t CRadixSortTask<DataType>::Resize(uint32_t nn)
{
	assert(nn <= Parameters::_NUM_MAX_INPUT_ELEMS);

    if (mOptions.verbose){
        std::cout << "Resizing to  " << nn << std::endl;
    }
    mNumberKeys = nn;
    return mRadixSortGPU.Resize(nn);
}


template <typename DataType>
void CRadixSortTask<DataType>::ExecuteTask(
        cl_context ,
        cl_command_queue CommandQueue,
        const std::array<size_t,3>& )
{
	assert(mNumberKeysRounded <= Parameters::_NUM_MAX_INPUT_ELEMS);
    if (mOptions.verbose) {
        std::cout << "Sorting " << mNumberKeys << " keys..." << std::endl;
    }
    mRadixSortGPU.calculate(CommandQueue);
    if (mOptions.verbose){
        std::cout << "Finished sorting." << std::endl;
    }
}

template <typename Stream>
void writePerformance(
    Stream&& stream,
    const RuntimesGPU& runtimesGPU,
    const RuntimesCPU& runtimesCPU,
    size_t numberKeys,
    const std::string& datasetName,
    const std::string& datatype

)
{
    const std::vector<std::string> columns {
        "NumElements", "Datatype", "Dataset", "avgHistogram", "avgScan", "avgPaste", "avgReorder", "avgTotalGPU", "avgTotalSTLCPU", "avgTotalRDXCPU"
    };

    stream << columns[0];
    for (auto i = 1U; i < columns.size(); i++) {
        stream << "," << columns[i];
    }

    const auto& timesGPU {runtimesGPU};
    const auto& timesCPU {runtimesCPU};

    stream << std::endl;
    stream << numberKeys << ",";
    stream << datatype << ",";
    stream << datasetName << ",";

    stream << timesGPU.timeHisto.avg << ",";
    stream << timesGPU.timeScan.avg << ",";
    stream << timesGPU.timePaste.avg << ",";
    stream << timesGPU.timeReorder.avg << ",";
    stream << timesGPU.timeTotal.avg << ",";

    stream << timesCPU.timeSTL.avg << ",";
    stream << timesCPU.timeRadix.avg;
    stream << std::endl;
}

template<class Callable>
void TestPerformance(
        cl_command_queue CommandQueue,
        Callable&& fun,
        const RadixSortOptions& options,
        const size_t numIterations,
        size_t numberKeys,
        const std::string& datasetName,
        const std::string& datatype
    )
{
    CTimer timer;
    timer.Start();

    decltype(fun()) lastMeasurements;

	for (auto i { 0U }; i < numIterations; i++) {
        lastMeasurements = fun();
    }

    //wait until the command queue is empty again
    V_RETURN_CL(clFinish(CommandQueue), "Error finishing the queue!");

    timer.Stop();
	double averageTimeTotal_ms = timer.GetElapsedMilliseconds() / double(numIterations);
    if (options.perf_to_stdout) {
        const auto& t{lastMeasurements.first};

        std::cout << " kernel |    avg      |     min     |    max " << std::endl;
        std::cout << " -----------------------------------------------" << std::endl;
        std::cout << "  histogram: " << std::setw(8) << t.timeHisto.avg << " | " << t.timeHisto.min << " | " << t.timeHisto.max << std::endl;
        std::cout << "  scan:      " << std::setw(8) << t.timeScan.avg << " | " << t.timeScan.min << " | " << t.timeScan.max << std::endl;
        std::cout << "  paste:     " << std::setw(8) << t.timePaste.avg << " | " << t.timePaste.min << " | " << t.timePaste.max << std::endl;
        std::cout << "  reorder:   " << std::setw(8) << t.timeReorder.avg << " | " << t.timeReorder.min << " | " << t.timeReorder.max << std::endl;
        std::cout << " -----------------------------------------------" << std::endl;
        std::cout << "  total:     " << averageTimeTotal_ms << " ms, throughput: " << 1.0e-6 * (double)numberKeys / averageTimeTotal_ms << " Gelem/s" << std::endl;
    }

    using std::chrono::system_clock;
    std::time_t tt = system_clock::to_time_t(system_clock::now());
    struct std::tm * ptm = std::localtime(&tt);
    const std::string dateFormat = "%H-%M-%S";
    std::stringstream fileNameBuilder;

    fileNameBuilder << "radix_" << std::put_time(ptm, dateFormat.c_str()) << ".csv";

    if (options.perf_to_csv) {
        const auto filename = fileNameBuilder.str();
        bool file_exists = false;

        {
            struct stat buffer;
            file_exists = (stat(filename.c_str(), &buffer) == 0);
        }

        // Print columns
        if (file_exists)
        {
            std::cout << "File " << filename << " already exists, not overwriting!" << std::endl;
        } else {
            std::ofstream outstream(filename, std::ofstream::out | std::ofstream::app);
            writePerformance(
                outstream,
                lastMeasurements.first,
                lastMeasurements.second,
                numberKeys,
                datasetName,
                datatype
            );
        }
    }
    if (options.perf_csv_to_stdout) {
        writePerformance(
            std::cout,
            lastMeasurements.first,
            lastMeasurements.second,
            numberKeys,
            datasetName,
            datatype
        );
    }
}

// Specialize CRadixSortTask for the supported types.
template class CRadixSortTask < int32_t >;
template class CRadixSortTask < int64_t >;
template class CRadixSortTask < uint32_t >;
template class CRadixSortTask < uint64_t >;

