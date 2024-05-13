#ifndef TENSORRT_ARGS_PARSER_H
#define TENSORRT_ARGS_PARSER_H

#ifdef _MSC_VER
#include "getOptWin.h"
#else
#include <getopt.h>
#endif
#include <iostream>
#include <string>
#include <vector>

namespace samplesCommon
{

//!
//! \brief The SampleParams structure groups the basic parameters required by
//!        all sample networks.
//!
struct SampleParams
{
    int32_t batchSize{1};              //!< Number of inputs in a batch
    int32_t dlaCore{-1};               //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    bool bf16{false};                  //!< Allow running the network in BF16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
};

//!
//! \brief The OnnxSampleParams structure groups the additional parameters required by
//!         networks that use ONNX
//!
struct OnnxSampleParams : public SampleParams
{
    std::string onnxFileName; //!< Filename of ONNX file of a network
};

//!
//! /brief Struct to maintain command-line arguments.
//!
struct Args
{
    bool runInInt8{false};
    bool runInFp16{false};
    bool runInBf16{false};
    bool help{false};
    int32_t useDLACore{-1};
    int32_t batch{1};
    std::vector<std::string> dataDirs;
    std::string saveEngine;
    std::string loadEngine;
    bool rowMajor{true};
};

//!
//! \brief Populates the Args struct with the provided command-line parameters.
//!
//! \throw invalid_argument if any of the arguments are not valid
//!
//! \return boolean If return value is true, execution can continue, otherwise program should exit
//!
inline bool parseArgs(Args& args, int32_t argc, char* argv[])
{
    while (1)
    {
        int32_t arg;
        static struct option long_options[] = {{"help", no_argument, nullptr, 'h'}, {"datadir", required_argument, nullptr, 'd'},
            {"int8", no_argument, nullptr, 'i'}, {"fp16", no_argument, nullptr, 'f'}, {"bf16", no_argument, nullptr, 'z'},
            {"columnMajor", no_argument, nullptr, 'c'}, {"saveEngine", required_argument, nullptr, 's'},
            {"loadEngine", required_argument, nullptr, 'o'}, {"useDLACore", required_argument, nullptr, 'u'},
            {"batch", required_argument, nullptr, 'b'}, {nullptr, 0, nullptr, 0}};
        int32_t option_index = 0;
        arg = getopt_long(argc, argv, "hd:iu", long_options, &option_index);
        if (arg == -1)
        {
            break;
        }

        switch (arg)
        {
        case 'h': args.help = true; return true;
        case 'd':
            if (optarg)
            {
                args.dataDirs.push_back(optarg);
            }
            else
            {
                std::cerr << "ERROR: --datadir requires option argument" << std::endl;
                return false;
            }
            break;
        case 's':
            if (optarg)
            {
                args.saveEngine = optarg;
            }
            break;
        case 'o':
            if (optarg)
            {
                args.loadEngine = optarg;
            }
            break;
        case 'i': args.runInInt8 = true; break;
        case 'f': args.runInFp16 = true; break;
        case 'z': args.runInBf16 = true; break;
        case 'c': args.rowMajor = false; break;
        case 'u':
            if (optarg)
            {
                args.useDLACore = std::stoi(optarg);
            }
            break;
        case 'b':
            if (optarg)
            {
                args.batch = std::stoi(optarg);
            }
            break;
        default: return false;
        }
    }
    return true;
}

} // namespace samplesCommon

#endif // TENSORRT_ARGS_PARSER_H
